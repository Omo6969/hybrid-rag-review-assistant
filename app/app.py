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


