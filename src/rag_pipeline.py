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
            "If the reviews do not contain enough information, say so.",
        ),
        ("human", "Review excerpts:\n{context}\n\nQuestion: {question}"),
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

def build_context(docs: list[Document]) -> str:
    """Format a list of LangChain Documents into a numbered context block.

    Args:
        docs: Retrieved LangChain Document objects.

    Returns:
        A structured, prompt-ready string.
    """
    parts = []
    for i, doc in enumerate(docs, 1):
        m = doc.metadata
        asin = m.get("parent_asin", "N/A")
        title = m.get("title", "")
        rating = m.get("rating", "N/A")
        text = doc.page_content[:400]
        parts.append(
            f"[{i}] ASIN: {asin} | Product: {title} | Rating: {rating}/5\n{text}"
        )
    return "\n\n".join(parts)

# Step 2 - Vectorstore builder

_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def build_semantic_vectorstore(
    documents: list[dict],
    model_name: str = _EMBED_MODEL,
) -> FAISS:
    """Build a LangChain FAISS vectorstore from a list of document dicts.

    Reuses the same embedding model as Milestone 1.

    Args:
        documents: List of dicts with at least a ``"text"`` key.
        model_name: Sentence-transformer model for embeddings.

    Returns:
        A LangChain FAISS vectorstore ready to be used as a retriever.
    """
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    lc_docs = []
    for i, doc in enumerate(documents):
        if not isinstance(doc, dict):
            raise ValueError(
                f"Invalid document at index {i}: expected dict, got "
                f"{type(doc).__name__}."
            )
        if "text" not in doc:
            raise ValueError(
                f"Invalid document at index {i}: missing required key 'text'."
            )
        lc_docs.append(
            Document(
                page_content=doc["text"],
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

    Args:
        retriever: Any LangChain-compatible retriever (semantic or hybrid).
        llm: An initialised ChatGroq instance.
        prompt_variant: One of ``"minimal"``, ``"concise"``, ``"detailed"``.

    Returns:
        An LCEL chain that accepts a query string and returns an answer string.
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
        """Generate an answer for *query*.

        Args:
            query: The user question.
            documents: Optional list of retrieved review dicts (must have a
                       ``"text"`` key). When provided, uses the RAG prompt.

        Returns:
            The model's answer as a plain string.
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
        lines = []
        for i, doc in enumerate(documents[:max_docs], 1):
            title = doc.get("title", "")
            text = doc.get("text", "")
            rating = doc.get("rating", "")
            snippet = f"{i}. [{title}] (Rating: {rating})\n   {text[:300]}"
            lines.append(snippet)
        return "\n\n".join(lines)
