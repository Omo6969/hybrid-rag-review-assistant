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


