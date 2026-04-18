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


