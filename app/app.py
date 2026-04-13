from pathlib import Path
import hashlib
import csv
from datetime import datetime

from shiny import App, ui, render

from src.bm25 import BM25Retriever
from src.semantic import SemanticRetriever


# Load retrievers
def load_retrievers():
    bm25 = None
    sem = None

    if Path("data/processed/bm25_index").exists():
        try:
            bm25 = BM25Retriever.load("data/processed/bm25_index")
        except:
            bm25 = None

    if Path("data/processed/semantic_index").exists():
        try:
            sem = SemanticRetriever.load("data/processed/semantic_index")
        except:
            sem = None

    return bm25, sem


# Helpers
def truncate(text, n=200):
    return text if len(text) <= n else text[: n - 3] + "..."


def get_doc_id(doc):
    text = (doc.get("title", "") + doc.get("text", "")).encode("utf-8")
    return hashlib.md5(text).hexdigest()


# UI 
app_ui = ui.page_fluid(
    ui.h2("🔍 Amazon Review Retrieval App"),
    ui.p("Search product reviews using BM25, Semantic, or Hybrid search."),

    ui.page_sidebar(
        ui.sidebar(
            ui.input_radio_buttons(
                "mode",
                "Search Mode",
                choices=["BM25", "Semantic", "Hybrid"],
                selected="BM25",
            ),
            ui.input_numeric(
                "top_k",
                "Number of results (default: 3)",
                value=3,
                min=1,
                max=10,
            ),
            ui.input_text("query", "Enter your query"),
            ui.input_action_button("search", "Search"),
        ),
        ui.output_ui("results"),
    ),
)


