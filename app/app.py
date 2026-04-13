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
        except Exception:
            bm25 = None

    if Path("data/processed/semantic_index").exists():
        try:
            sem = SemanticRetriever.load("data/processed/semantic_index")
        except Exception:
            sem = None

    return bm25, sem


# Helpers
def truncate(text, n=200):
    return text if len(text) <= n else text[: n - 3] + "..."


def get_doc_id(doc):
    """
    Compute a document identifier from a retrieval document.

    If the document has a "doc_id" field, it is used directly.
    Otherwise, a hash of the concatenation of the title and text fields is used.

    Returns:
        str: The document identifier.
    """
    if doc.get("doc_id") is not None:
        return str(doc["doc_id"])
    text = (doc.get("title", "") + doc.get("text", "")).encode("utf-8")
    return hashlib.md5(text).hexdigest()


# UI
app_ui = ui.page_fluid(
    ui.h2("🔍 Amazon Review Retrieval App"),
    ui.p("Search Amazon product reviews using BM25, semantic, or hybrid retrieval."),

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
                "Number of results",
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


# Server
def server(input, output, session):
    bm25, sem = load_retrievers()

    # Feedback file setup
    feedback_file = Path("data/processed/feedback.csv")
    feedback_file.parent.mkdir(parents=True, exist_ok=True)

    if not feedback_file.exists():
        with open(feedback_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["timestamp", "query", "mode", "doc_id", "title", "score", "feedback"]
            )

    prev_counts = {}

    @output
    @render.ui
    def results():
        if input.search() == 0:
            return ui.p("Enter a query and click Search.")

        query = input.query().strip()
        mode = input.mode()
        top_k = int(input.top_k() or 3)

        if not query:
            return ui.p("Please enter a query.")

        # Retrieval
        if mode == "BM25":
            if bm25 is None:
                return ui.p("BM25 index not found. Please build it first.")
            results_docs = bm25.search(query, top_k)

        elif mode == "Semantic":
            if sem is None:
                return ui.p("Semantic index not found. Please build it first.")
            results_docs = sem.search(query, top_k)

        else:  # Hybrid
            if bm25 is None or sem is None:
                return ui.p("Both BM25 and Semantic indices are required.")

            bm25_res = bm25.search(query, top_k * 5)
            sem_res = sem.search(query, top_k * 5)

            scores = {}
            docs = {}

            for r in bm25_res:
                doc_id = get_doc_id(r)
                scores[doc_id] = scores.get(doc_id, 0) + float(r.get("score", 0))
                docs[doc_id] = r

            for r in sem_res:
                doc_id = get_doc_id(r)
                scores[doc_id] = scores.get(doc_id, 0) + float(r.get("score", 0))
                docs[doc_id] = r

            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

            results_docs = []
            for doc_id, score in ranked:
                doc = docs[doc_id]
                doc["score"] = score
                results_docs.append(doc)

        # Display
        ui_list = []

        for i, r in enumerate(results_docs, 1):
            title = r.get("title", "No title")
            text = truncate(r.get("text", ""))
            rating = r.get("rating", "N/A")
            score = float(r.get("score", 0))

            doc_id = get_doc_id(r)

            try:
                stars = "★" * int(round(float(rating)))
            except Exception:
                stars = str(rating)

            like_id = f"like_{doc_id}"
            dislike_id = f"dislike_{doc_id}"

            ui_list.append(
                ui.div(
                    ui.h5(f"{i}. {title}"),
                    ui.p(text),
                    ui.p(f"Rating: {stars} ({rating})"),
                    ui.p(f"Score: {score:.4f}"),
                    ui.input_action_button(like_id, "👍"),
                    ui.input_action_button(dislike_id, "👎"),
                    ui.hr(),
                )
            )

            # Save feedback
            for btn_id, value in [(like_id, 1), (dislike_id, -1)]:
                try:
                    current = int(getattr(input, btn_id)())
                except Exception:
                    current = 0

                prev = prev_counts.get(btn_id, 0)

                if current > prev:
                    with open(feedback_file, "a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow(
                            [
                                datetime.utcnow().isoformat(),
                                query,
                                mode,
                                doc_id,
                                title,
                                f"{score:.6f}",
                                value,
                            ]
                        )

                    prev_counts[btn_id] = current

        if not ui_list:
            return ui.p("No results found.")

        return ui.TagList(ui_list)


# Run App
app = App(app_ui, server)
