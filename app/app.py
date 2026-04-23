
from __future__ import annotations

import csv
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any

from shiny import App, render, ui

from src.bm25 import BM25Retriever
from src.hybrid import HybridRetriever
from src.rag_pipeline import LLMPipeline
from src.semantic import SemanticRetriever
import os

DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


def load_retrievers() -> tuple[
    BM25Retriever | None,
    SemanticRetriever | None,
    HybridRetriever | None,
]:
    """Load the retrievers required by the app.

    Returns
    -------
    tuple
        Tuple containing the BM25 retriever, semantic retriever, and hybrid
        retriever. Any component that cannot be loaded is returned as ``None``.
    """
    bm25 = None
    semantic = None
    hybrid = None

    bm25_path = Path("data/processed/bm25_index")
    semantic_path = Path("data/processed/semantic_index")

    if bm25_path.exists():
        try:
            bm25 = BM25Retriever.load(bm25_path)
        except Exception:
            bm25 = None

    if semantic_path.exists():
        try:
            semantic = SemanticRetriever.load(semantic_path)
        except Exception:
            semantic = None

    if bm25 is not None and semantic is not None:
        try:
            hybrid = HybridRetriever(
                bm25_retriever=bm25,
                semantic_retriever=semantic,
                bm25_weight=0.4,
                semantic_weight=0.6,
                rrf_k=60,
                fetch_k=10,
                key_field="doc_id",
            )
        except Exception:
            hybrid = None

    return bm25, semantic, hybrid


def load_llm_pipeline() -> LLMPipeline | None:
    """Load the RAG LLM pipeline.

    Returns
    -------
    LLMPipeline or None
        Instantiated LLM pipeline if available, otherwise ``None``.
    """
    try:
        return LLMPipeline(model=DEFAULT_MODEL, temperature=0.0)
    except Exception:
        return None


def truncate(text: str, n: int = 200) -> str:
    """Truncate text for display.

    Parameters
    ----------
    text : str
        Input text.
    n : int, default=200
        Maximum length of the returned string.

    Returns
    -------
    str
        Truncated text string.
    """
    return text if len(text) <= n else text[: n - 3] + "..."


def get_doc_id(doc: dict[str, Any]) -> str:
    """Compute a stable identifier for a retrieval document.

    Parameters
    ----------
    doc : dict of str to Any
        Retrieval document.

    Returns
    -------
    str
        Document identifier.
    """
    if doc.get("doc_id") is not None:
        return str(doc["doc_id"])

    text = (doc.get("title", "") + doc.get("text", "")).encode("utf-8")
    return hashlib.md5(text).hexdigest()


def run_search_only(
    query: str,
    mode: str,
    top_k: int,
    bm25: BM25Retriever | None,
    semantic: SemanticRetriever | None,
    hybrid: HybridRetriever | None,
) -> tuple[list[dict[str, Any]], str | None]:
    """Run retrieval-only search.

    Parameters
    ----------
    query : str
        User query.
    mode : str
        Retrieval mode. One of ``BM25``, ``Semantic``, or ``Hybrid``.
    top_k : int
        Number of results to return.
    bm25 : BM25Retriever or None
        BM25 retriever.
    semantic : SemanticRetriever or None
        Semantic retriever.
    hybrid : HybridRetriever or None
        Hybrid retriever.

    Returns
    -------
    tuple
        Tuple of ``(results, error_message)``. If successful, the error message
        is ``None``.
    """
    if mode == "BM25":
        if bm25 is None:
            return [], "BM25 index not found. Please build it first."
        return bm25.search(query, top_k), None

    if mode == "Semantic":
        if semantic is None:
            return [], "Semantic index not found. Please build it first."
        return semantic.search(query, top_k), None

    if hybrid is None:
        return [], "Both BM25 and Semantic indices are required for Hybrid mode."

    return hybrid.search(query, top_k), None


def run_rag(
    query: str,
    mode: str,
    top_k: int,
    semantic: SemanticRetriever | None,
    hybrid: HybridRetriever | None,
    llm_pipeline: LLMPipeline | None,
) -> tuple[str | None, list[dict[str, Any]], str | None]:
    """Run RAG mode.

    Parameters
    ----------
    query : str
        User query.
    mode : str
        RAG mode. One of ``Semantic RAG`` or ``Hybrid RAG``.
    top_k : int
        Number of retrieved documents to use.
    semantic : SemanticRetriever or None
        Semantic retriever.
    hybrid : HybridRetriever or None
        Hybrid retriever.
    llm_pipeline : LLMPipeline or None
        LLM pipeline for grounded generation.

    Returns
    -------
    tuple
        Tuple of ``(answer, docs, error_message)``.
    """
    if llm_pipeline is None:
        return None, [], (
            "RAG mode is unavailable because the LLM pipeline could not be "
            "initialized. Check that your GROQ_API_KEY is set."
        )

    if mode == "Semantic RAG":
        if semantic is None:
            return None, [], "Semantic index not found. Please build it first."
        docs = semantic.search(query, top_k)
    else:
        if hybrid is None:
            return None, [], (
                "Hybrid RAG requires both BM25 and Semantic indices to be available."
            )
        docs = hybrid.search(query, top_k)

    answer = llm_pipeline.generate(query=query, documents=docs)
    return answer, docs, None


def render_result_cards(
    results_docs: list[dict[str, Any]],
    query: str,
    mode: str,
    input: Any,
    feedback_file: Path,
    prev_counts: dict[str, int],
) -> list[Any]:
    """Render retrieval results as UI cards and log user feedback.

    Parameters
    ----------
    results_docs : list of dict of str to Any
        Retrieved documents to display.
    query : str
        User query.
    mode : str
        Current app mode or retriever mode.
    input : Any
        Shiny input object.
    feedback_file : pathlib.Path
        Feedback CSV file path.
    prev_counts : dict of str to int
        Button click counter cache to avoid duplicate logging.

    Returns
    -------
    list
        List of Shiny UI components.
    """
    ui_list: list[Any] = []

    for i, result in enumerate(results_docs, 1):
        title = result.get("title", "No title")
        text = truncate(result.get("text", ""))
        rating = result.get("rating", "N/A")
        score = float(result.get("score", 0))
        doc_id = get_doc_id(result)

        try:
            stars = "★" * int(round(float(rating)))
        except Exception:
            stars = str(rating)

        retrieval_sources = result.get("retrieval_sources")
        sources_line = (
            f"Sources: {', '.join(retrieval_sources)}"
            if isinstance(retrieval_sources, list) and retrieval_sources
            else None
        )

        like_id = f"like_{doc_id}"
        dislike_id = f"dislike_{doc_id}"

        children = [
            ui.h5(f"{i}. {title}"),
            ui.p(text),
            ui.p(f"Rating: {stars} ({rating})"),
            ui.p(f"Score: {score:.4f}"),
        ]

        if result.get("parent_asin"):
            children.append(ui.p(f"ASIN: {result['parent_asin']}"))

        if sources_line is not None:
            children.append(ui.p(sources_line))

        children.extend(
            [
                ui.input_action_button(like_id, "👍"),
                ui.input_action_button(dislike_id, "👎"),
                ui.hr(),
            ]
        )

        ui_list.append(ui.div(*children, class_="result-card"))

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

    return ui_list


app_ui = ui.page_fluid(
    ui.tags.style("""
        body {
            background-color: #f8fafc;
            color: #1f2937;
            font-family: Arial, sans-serif;
        }

        .container-fluid {
            max-width: 1200px;
            margin: 0 auto;
            padding-top: 1rem;
            padding-bottom: 2rem;
        }

        h2 {
            font-weight: 700;
            margin-bottom: 0.4rem;
            background: #e0f2fe;
            padding: 0.9rem 1rem;
            border-radius: 14px;
            border: 1px solid #bae6fd;
        }

        .muted-text {
            color: #6b7280;
            font-size: 0.95rem;
            margin-top: 0.6rem;
            margin-bottom: 1rem;
        }

        .sidebar {
            background: #f0f9ff;
            border: 1px solid #bae6fd;
            border-radius: 14px;
            padding: 1rem;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        }

        .shiny-input-radiogroup > label,
        .shiny-input-container > label {
            display: block;
            font-weight: 600;
            color: #111827;
            margin-bottom: 0.7rem !important;
        }

        .radio {
            margin-top: 0.35rem;
            margin-bottom: 1.1rem;
        }

        .radio label {
            display: block;
            margin-bottom: 0.45rem;
        }

        .form-control,
        .form-select,
        .btn {
            border-radius: 10px !important;
        }

        .btn {
            font-weight: 600;
        }

        .result-card {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 1rem 1.2rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        }

        .answer-panel {
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            border-radius: 12px;
            padding: 1rem 1.2rem;
            margin-bottom: 1rem;
        }

        hr {
            margin-top: 1rem;
            margin-bottom: 0;
            border-top: 1px solid #e5e7eb;
        }
    """),

    ui.h2("🔍 Amazon Product Query Assistant"),
    ui.p(
        "Search Amazon product reviews with BM25, semantic, or hybrid retrieval, "
        "and switch to RAG mode for grounded answer generation.",
        class_="muted-text",
    ),
    ui.page_sidebar(
        ui.sidebar(
            ui.input_radio_buttons(
                "app_mode",
                "App Mode",
                choices=["Search Only", "RAG Mode"],
                selected="Search Only",
            ),
            ui.output_ui("mode_selector"),
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


def server(input, output, session):
    """Run the Shiny server for the Milestone 2 app.

    Parameters
    ----------
    input : Any
        Shiny input object.
    output : Any
        Shiny output object.
    session : Any
        Shiny session object.

    Returns
    -------
    None
    """
    bm25, semantic, hybrid = load_retrievers()
    llm_pipeline = load_llm_pipeline()

    feedback_file = Path("data/processed/feedback.csv")
    feedback_file.parent.mkdir(parents=True, exist_ok=True)

    if not feedback_file.exists():
        with open(feedback_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["timestamp", "query", "mode", "doc_id", "title", "score", "feedback"]
            )

    prev_counts: dict[str, int] = {}

    @output
    @render.ui
    def mode_selector():
        """Render the mode selector based on the chosen app mode.

        Returns
        -------
        Any
            Shiny UI component.
        """
        if input.app_mode() == "Search Only":
            return ui.input_radio_buttons(
                "retrieval_mode",
                "Search Mode",
                choices=["BM25", "Semantic", "Hybrid"],
                selected="BM25",
            )

        return ui.input_radio_buttons(
            "retrieval_mode",
            "RAG Mode",
            choices=["Semantic RAG", "Hybrid RAG"],
            selected="Hybrid RAG",
        )

    @output
    @render.ui
    def results():
        """Render the app results panel.

        Returns
        -------
        Any
            Shiny UI output.
        """
        if input.search() == 0:
            return ui.TagList(
                ui.p("Enter a query and click Search."),
                ui.p(
                    "Use Search Only for retrieval results or RAG Mode for a generated answer "
                    "grounded in retrieved review documents."
                ),
            )

        query = input.query().strip()
        app_mode = input.app_mode()
        mode = input.retrieval_mode() or "BM25"
        top_k = int(input.top_k() or 3)

        if not query:
            return ui.p("Please enter a query.")

        if app_mode == "Search Only":
            results_docs, error = run_search_only(
                query=query,
                mode=mode,
                top_k=top_k,
                bm25=bm25,
                semantic=semantic,
                hybrid=hybrid,
            )

            if error is not None:
                return ui.p(error)

            cards = render_result_cards(
                results_docs=results_docs,
                query=query,
                mode=mode,
                input=input,
                feedback_file=feedback_file,
                prev_counts=prev_counts,
            )

            if not cards:
                return ui.p("No results found.")

            return ui.TagList(
                ui.h4("Retrieved Results"),
                *cards,
            )

        answer, results_docs, error = run_rag(
            query=query,
            mode=mode,
            top_k=top_k,
            semantic=semantic,
            hybrid=hybrid,
            llm_pipeline=llm_pipeline,
        )

        if error is not None:
            return ui.p(error)

        cards = render_result_cards(
            results_docs=results_docs,
            query=query,
            mode=mode,
            input=input,
            feedback_file=feedback_file,
            prev_counts=prev_counts,
        )

        answer_panel = ui.div(
            ui.h4("Generated Answer"),
            ui.p(answer or "No answer generated."),
            class_="answer-panel",
        )

        if not cards:
            return ui.TagList(
                answer_panel,
                ui.p("No supporting documents found."),
            )

        return ui.TagList(
            answer_panel,
            ui.h4("Supporting Retrieved Documents"),
            *cards,
        )


app = App(app_ui, server)
