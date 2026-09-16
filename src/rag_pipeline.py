"""
RAG pipeline for the Amazon Product Query Assistant.

Provides:
  - LLMPipeline  : thin Groq/Llama wrapper for plain and dict-based RAG (Step 1)
  - build_semantic_vectorstore : LangChain FAISS vectorstore from document dicts (Step 2)
  - build_context              : format LangChain Documents into a prompt context (Step 2)
  - PROMPT_VARIANTS            : three prompt templates to experiment with (Step 2)
  - build_rag_chain            : LCEL chain wiring retriever -> context -> prompt -> LLM (Step 2/3)
"""

from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# Step 1 – plain LLM templates 

_PLAIN_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that answers questions about Amazon products "
            "based on customer reviews. Be concise and factual.",
        ),
        ("human", "{question}"),
    ]
)

_RAG_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that answers questions about Amazon products "
            "based on customer reviews. Use only the provided review excerpts to answer. "
            "The context below is grouped by product: each product's name is followed by "
            "the customer reviews for it. When you refer to a product, always use its "
            "product name (e.g. \"Product: ...\") -- never refer to a review by number "
            "(e.g. never say \"review 1\" or \"review #4\"). "
            "If the reviews do not contain enough information, say so.",
        ),
        ("human", "Product reviews:\n{context}\n\nQuestion: {question}"),
    ]
)

# Step 2 – Prompt variants for the LCEL RAG chain
PROMPT_VARIANTS: dict[str, ChatPromptTemplate] = {
    # Variant 1 - minimal: shortest possible instruction
    "minimal": ChatPromptTemplate.from_messages(
        [
            ("system", "Answer using only the provided reviews. Be brief."),
            ("human", "Reviews:\n{context}\n\nQ: {question}\nA:"),
        ]
    ),
    # Variant 2 - concise shopping assistant (recommended default)
    "concise": ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful Amazon shopping assistant.\n"
                "Answer the question using ONLY the following product reviews.\n"
                "Be concise and factual. If the reviews do not answer the question, say so.",
            ),
            ("human", "Reviews:\n{context}\n\nQuestion: {question}"),
        ]
    ),
    # Variant 3 - detailed with citation encouragement
    "detailed": ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert Amazon product reviewer assistant.\n"
                "Answer the question using ONLY the provided review excerpts.\n"
                "Always cite the product ASIN when possible.\n"
                "Mention both positive and negative aspects found in the reviews.\n"
                "If the reviews do not contain enough information, clearly state that.",
            ),
            (
                "human",
                "Product Reviews:\n{context}\n\nShopper Question: {question}\n\nAnswer:",
            ),
        ]
    ),
}


# Step 2 - Context builder

def _resolve_product_name(fields: dict[str, object]) -> str:
    """Resolve a shopper-friendly product name for a retrieval document.

    Prefers the product's real name (``product_title``, sourced from item
    metadata during preprocessing) over the review's own headline
    (``title`` -- user-written text like "This stuff is your friend!") or a
    bare ASIN, so the assistant can refer to products by name instead of an
    opaque code or a review index.

    Parameters
    ----------
    fields : dict of str to object
        A retrieval document (or LangChain ``Document.metadata``).

    Returns
    -------
    str
        The best available product name.
    """
    product_title = str(fields.get("product_title") or "").strip()
    if product_title:
        return product_title

    review_title = str(fields.get("title") or "").strip()
    if review_title:
        return review_title

    return f"Product {fields.get('parent_asin', 'unknown')}"


def _group_context_blocks(entries: list[tuple[str, object, str]]) -> str:
    """Format ``(product_name, rating, text)`` entries into a context block.

    Reviews for the same product are grouped together under one "Product:"
    heading, rather than emitted as a flat, independently numbered list --
    this is what lets the model talk about a named product supported by one
    or more reviews, instead of citing "review 1" / "review #4".

    Parameters
    ----------
    entries : list of tuple of (str, object, str)
        ``(product_name, rating, review_text)`` tuples, one per review.

    Returns
    -------
    str
        A structured, prompt-ready string.
    """
    grouped: dict[str, list[tuple[object, str]]] = {}
    order: list[str] = []

    for product_name, rating, text in entries:
        if product_name not in grouped:
            grouped[product_name] = []
            order.append(product_name)
        grouped[product_name].append((rating, text))

    blocks = []
    for product_name in order:
        review_lines = [
            f"  - (Rating: {rating}/5) {text}" for rating, text in grouped[product_name]
        ]
        blocks.append(f"Product: {product_name}\n" + "\n".join(review_lines))

    return "\n\n".join(blocks)


def build_context(docs: list[Document]) -> str:
    """Format a list of LangChain Documents into a product-grouped context.

    Args:
        docs: Retrieved LangChain Document objects.

    Returns:
        A structured, prompt-ready string, grouped by product name.
    """
    entries = [
        (_resolve_product_name(doc.metadata), doc.metadata.get("rating", "N/A"), doc.page_content[:400])
        for doc in docs
    ]
    return _group_context_blocks(entries)


# Step 2 - Vectorstore builder

_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def build_semantic_vectorstore(
    documents: list[dict[str, object]],
    model_name: str = _EMBED_MODEL,
) -> FAISS:
    """Build a LangChain FAISS vector store from retrieval document dictionaries.

    Parameters
    ----------
    documents : list of dict of str to object
        Retrieval documents. Each document must include a ``"text"`` key.
    model_name : str, default="sentence-transformers/all-MiniLM-L6-v2"
        Sentence-transformer model name used for embeddings.

    Returns
    -------
    langchain_community.vectorstores.FAISS
        A LangChain FAISS vector store ready for use as a retriever.

    Raises
    ------
    ValueError
        If a document is not a dictionary or does not contain a ``"text"`` key.
    """
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    lc_docs: list[Document] = []

    for i, doc in enumerate(documents):
        if not isinstance(doc, dict):
            raise ValueError(
                f"Invalid document at index {i}: expected dict, got {type(doc).__name__}."
            )
        if "text" not in doc:
            raise ValueError(
                f"Invalid document at index {i}: missing required key 'text'."
            )

        lc_docs.append(
            Document(
                page_content=str(doc["text"]),
                metadata={k: v for k, v in doc.items() if k != "text"},
            )
        )

    return FAISS.from_documents(lc_docs, embeddings)


# Step 2 – LCEL RAG chain builder

def build_rag_chain(
    retriever,
    llm: ChatGroq,
    prompt_variant: str = "concise",
):
    """Build an LCEL RAG chain.

    Parameters
    ----------
    retriever : Any
        LangChain-compatible retriever that returns LangChain documents.
    llm : ChatGroq
        Initialised Groq chat model.
    prompt_variant : str, default="concise"
        Prompt variant name. Must be one of ``"minimal"``, ``"concise"``,
        or ``"detailed"``.

    Returns
    -------
    Any
        An LCEL chain that accepts a query string and returns an answer string.

    Raises
    ------
    ValueError
        If ``prompt_variant`` is not one of the supported prompt variants.
    """
    if prompt_variant not in PROMPT_VARIANTS:
        raise ValueError(
            f"Unknown prompt_variant '{prompt_variant}'. "
            f"Choose from: {list(PROMPT_VARIANTS)}"
        )

    prompt = PROMPT_VARIANTS[prompt_variant]

    return (
        {"context": retriever | build_context, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


# Step 1 – LLMPipeline (plain + dict-based RAG, kept for backward compat)


class LLMPipeline:
    """Wraps a Groq-hosted Llama 3 model for plain and RAG generation.

    This is the Step 1 interface. For the full LCEL pipeline use
    ``build_rag_chain()`` instead.
    """

    def __init__(
        self,
        model: str = "llama-3.1-8b-instant",
        temperature: float = 0.0,
        max_tokens: int = 512,
        api_key: Optional[str] = None,
    ) -> None:
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise ValueError(
                "GROQ_API_KEY not found. Set it in your .env file or pass api_key=."
            )
        self.llm = ChatGroq(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=key,
        )
        self.model = model

    def generate(self, query: str, documents: Optional[list[dict]] = None) -> str:
        """Generate an answer for a user query.

        Parameters
        ----------
        query : str
            The user question.
        documents : list of dict or None, default=None
            Optional retrieved review documents. When provided, generation is
            grounded in the supplied context.

        Returns
        -------
        str
            The generated answer as a plain string.
        """
        if documents:
            context = self._build_context(documents)
            chain = _RAG_TEMPLATE | self.llm
            response = chain.invoke({"context": context, "question": query})
        else:
            chain = _PLAIN_TEMPLATE | self.llm
            response = chain.invoke({"question": query})

        return response.content.strip()

    @staticmethod
    def _build_context(documents: list[dict], max_docs: int = 5) -> str:
        """Format retrieved review documents into a product-grouped context.

        Parameters
        ----------
        documents : list of dict
            Retrieved documents, each expected to have ``text`` and
            ``rating``, plus (when available) ``product_title`` for the
            product's real name.
        max_docs : int, default=5
            Maximum number of documents to include.

        Returns
        -------
        str
            A structured, prompt-ready string, grouped by product name so
            the model can refer to products rather than review numbers.
        """
        entries = [
            (_resolve_product_name(doc), doc.get("rating", "N/A"), str(doc.get("text", ""))[:300])
            for doc in documents[:max_docs]
        ]
        return _group_context_blocks(entries)
