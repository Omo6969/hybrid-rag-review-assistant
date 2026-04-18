"""
LLM pipeline for the Amazon Product Query Assistant.

Provides a thin wrapper around ChatGroq (Llama 3) that supports:
  - plain generation (no context)
  - RAG generation (with retrieved document context)
"""

from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about Amazon products "
    "based on customer reviews. Be concise and factual."
)

_RAG_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about Amazon products "
    "based on customer reviews. Use only the provided review excerpts to answer. "
    "If the reviews do not contain enough information, say so."
)

_RAG_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", _RAG_SYSTEM_PROMPT),
        (
            "human",
            "Review excerpts:\n{context}\n\nQuestion: {question}",
        ),
    ]
)

_PLAIN_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)


class LLMPipeline:
    """Wraps a Groq-hosted Llama 3 model for plain and RAG generation."""

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

    # Public API

    def generate(self, query: str, documents: Optional[list[dict]] = None) -> str:
        """Generate an answer for *query*.

        Args:
            query: The user question.
            documents: Optional list of retrieved review dicts (must have a
                       "text" key).  When provided, uses RAG prompt.

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

    # Helpers
    @staticmethod
    def _build_context(documents: list[dict], max_docs: int = 5) -> str:
        """Format retrieved documents into a numbered context block."""
        lines = []
        for i, doc in enumerate(documents[:max_docs], 1):
            title = doc.get("title", "")
            text = doc.get("text", "")
            rating = doc.get("rating", "")
            snippet = f"{i}. [{title}] (Rating: {rating})\n   {text[:300]}"
            lines.append(snippet)
        return "\n\n".join(lines)


