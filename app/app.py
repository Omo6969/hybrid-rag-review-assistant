
from __future__ import annotations

import csv
import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from shiny import App, render, ui

from src.bm25 import BM25Retriever
from src.hybrid import HybridRetriever
from src.rag_pipeline import LLMPipeline
from src.semantic import SemanticRetriever

DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Number of supporting reviews the assistant grounds each answer in. This is
# an internal retrieval setting, not something a shopper needs to think
# about, so it isn't exposed as a UI control.
TOP_K = 5


def load_hybrid_retriever() -> HybridRetriever | None:
    """Load the hybrid (BM25 + semantic) retriever used by the assistant.

    The assistant always searches with hybrid retrieval -- it combines
    keyword and semantic matching under the hood -- so this is the only
    retriever the app needs to load.

    Returns
    -------
    HybridRetriever or None
        The hybrid retriever, or ``None`` if its underlying indices are
        unavailable or fail to load.
    """
    # Prefer the full locally-built indices (data/processed/). If they are not
    # present -- e.g. on a fresh deployment such as Posit Connect Cloud, where
    # the full 701k-row dataset isn't downloaded -- fall back to the small
    # prebuilt sample indices bundled at deploy/sample_data/ so the app still
    # works out of the box. See deploy/README.md for details.
    bm25_path = Path("data/processed/bm25_index")
    semantic_path = Path("data/processed/semantic_index")

    if not bm25_path.exists() and not semantic_path.exists():
        sample_bm25_path = Path("deploy/sample_data/bm25_index")
        sample_semantic_path = Path("deploy/sample_data/semantic_index")
        if sample_bm25_path.exists() and sample_semantic_path.exists():
            bm25_path = sample_bm25_path
            semantic_path = sample_semantic_path

    try:
        bm25 = BM25Retriever.load(bm25_path)
        semantic = SemanticRetriever.load(semantic_path)
        return HybridRetriever(
            bm25_retriever=bm25,
            semantic_retriever=semantic,
            bm25_weight=0.4,
            semantic_weight=0.6,
            rrf_k=60,
            fetch_k=10,
            key_field="doc_id",
        )
    except Exception:
        return None


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


def truncate(text: str, n: int = 220) -> str:
    """Truncate text for display.

    Parameters
    ----------
    text : str
        Input text.
    n : int, default=220
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


def ask_assistant(
    query: str,
    hybrid: HybridRetriever | None,
    llm_pipeline: LLMPipeline | None,
) -> tuple[str | None, list[dict[str, Any]], str | None]:
    """Answer a shopper's question, grounded in retrieved customer reviews.

    Parameters
    ----------
    query : str
        The shopper's question.
    hybrid : HybridRetriever or None
        Hybrid retriever used to find supporting reviews.
    llm_pipeline : LLMPipeline or None
        LLM pipeline used to generate the grounded answer.

    Returns
    -------
    tuple
        Tuple of ``(answer, supporting_reviews, error_message)``.
    """
    if hybrid is None:
        return None, [], (
            "The assistant isn't set up yet -- its review index hasn't been "
            "built. Run the setup steps in the README, then restart the app."
        )

    if llm_pipeline is None:
        return None, [], (
            "The assistant can't generate answers right now because its "
            "language model isn't configured. Check that GROQ_API_KEY is set."
        )

    docs = hybrid.search(query, TOP_K)
    answer = llm_pipeline.generate(query=query, documents=docs)
    return answer, docs, None


def render_review_cards(
    results_docs: list[dict[str, Any]],
    query: str,
    input: Any,
    feedback_file: Path,
    prev_counts: dict[str, int],
) -> list[Any]:
    """Render supporting reviews as compact cards and log user feedback.

    Only shopper-relevant details are shown -- product, rating, and a short
    quote from the review -- with no retrieval scores or internals.

    Parameters
    ----------
    results_docs : list of dict of str to Any
        Retrieved documents to display.
    query : str
        The shopper's question.
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

    for result in results_docs:
        title = result.get("title", "No title")
        text = truncate(result.get("text", ""))
        rating = result.get("rating", "N/A")
        score = float(result.get("score", 0))
        doc_id = get_doc_id(result)

        try:
            stars = "★" * int(round(float(rating))) + "☆" * (5 - int(round(float(rating))))
        except Exception:
            stars = str(rating)

        like_id = f"like_{doc_id}"
        dislike_id = f"dislike_{doc_id}"

        children = [
            ui.div(
                ui.span(title, class_="review-title"),
                ui.span(stars, class_="review-stars"),
                class_="review-card-header",
            ),
            ui.p(f"“{text}”", class_="review-quote"),
            ui.div(
                ui.span("Helpful?", class_="feedback-label"),
                ui.input_action_button(like_id, "👍", class_="feedback-btn"),
                ui.input_action_button(dislike_id, "👎", class_="feedback-btn"),
                class_="review-footer",
            ),
        ]

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
                            datetime.now(timezone.utc).isoformat(),
                            query,
                            "Hybrid RAG",
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
            background: linear-gradient(180deg, #f0f9ff 0%, #f8fafc 320px);
            color: #1f2937;
            font-family: 'Segoe UI', Arial, sans-serif;
        }

        .container-fluid {
            max-width: 760px;
            margin: 0 auto;
            padding-top: 2.5rem;
            padding-bottom: 3rem;
        }

        .hero {
            text-align: center;
            margin-bottom: 1.75rem;
        }

        .hero-icon {
            font-size: 2.4rem;
            line-height: 1;
        }

        .hero h2 {
            font-weight: 700;
            font-size: 1.6rem;
            margin: 0.4rem 0 0.3rem 0;
        }

        .hero p {
            color: #6b7280;
            font-size: 1rem;
            margin: 0;
        }

        .ask-bar {
            display: flex;
            gap: 0.6rem;
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 999px;
            padding: 0.5rem 0.5rem 0.5rem 1.2rem;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
            margin-bottom: 1.75rem;
            align-items: center;
        }

        .ask-bar .shiny-input-container,
        .ask-bar .form-group {
            flex: 1;
            margin-bottom: 0 !important;
        }

        .ask-bar input[type="text"] {
            border: none !important;
            box-shadow: none !important;
            padding: 0.4rem 0 !important;
            font-size: 1rem;
        }

        .ask-bar input[type="text"]:focus {
            outline: none;
            box-shadow: none;
        }

        #ask {
            border-radius: 999px !important;
            padding: 0.55rem 1.4rem !important;
            font-weight: 600;
            background-color: #0284c7;
            border-color: #0284c7;
            color: white;
            white-space: nowrap;
        }

        #ask:hover {
            background-color: #0369a1;
            border-color: #0369a1;
        }

        .hint-text {
            text-align: center;
            color: #9ca3af;
            font-size: 0.9rem;
            margin-top: -1rem;
            margin-bottom: 1.5rem;
        }

        .assistant-message {
            display: flex;
            gap: 0.8rem;
            background: #ffffff;
            border: 1px solid #e0f2fe;
            border-radius: 16px;
            padding: 1.1rem 1.3rem;
            margin-bottom: 1.6rem;
            box-shadow: 0 2px 10px rgba(15, 23, 42, 0.05);
        }

        .assistant-avatar {
            font-size: 1.5rem;
            line-height: 1.4;
        }

        .assistant-text {
            margin: 0;
            line-height: 1.55;
            white-space: pre-wrap;
        }

        .evidence-heading {
            font-size: 0.95rem;
            font-weight: 700;
            color: #6b7280;
            text-transform: uppercase;
            letter-spacing: 0.03em;
            margin-bottom: 0.8rem;
        }

        .result-card {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 0.9rem 1.1rem;
            margin-bottom: 0.9rem;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        }

        .review-card-header {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            gap: 0.5rem;
            margin-bottom: 0.35rem;
        }

        .review-title {
            font-weight: 600;
            color: #111827;
        }

        .review-stars {
            color: #f59e0b;
            font-size: 0.9rem;
            white-space: nowrap;
        }

        .review-quote {
            color: #4b5563;
            font-style: italic;
            margin: 0 0 0.5rem 0;
        }

        .review-footer {
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }

        .feedback-label {
            color: #9ca3af;
            font-size: 0.82rem;
            margin-right: 0.2rem;
        }

        .feedback-btn {
            border: none !important;
            background: transparent !important;
            padding: 0.1rem 0.3rem !important;
            font-size: 0.95rem;
        }

        .empty-state {
            text-align: center;
            color: #9ca3af;
            margin-top: 1.5rem;
        }

        .error-banner {
            background: #fef2f2;
            border: 1px solid #fecaca;
            color: #991b1b;
            border-radius: 12px;
            padding: 0.9rem 1.1rem;
        }
    """),

    ui.tags.script("""
        document.addEventListener('keydown', function (event) {
            if (event.key === 'Enter' && document.activeElement && document.activeElement.id === 'query') {
                event.preventDefault();
                var askButton = document.getElementById('ask');
                if (askButton) { askButton.click(); }
            }
        });
    """),

    ui.div(
        ui.div("🛍️", class_="hero-icon"),
        ui.h2("Amazon Beauty Shopping Assistant"),
        ui.p("Ask a question and get an answer grounded in real customer reviews."),
        class_="hero",
    ),

    ui.div(
        ui.input_text(
            "query",
            None,
            placeholder="e.g. What's a good lip balm for very dry lips?",
            width="100%",
        ),
        ui.input_action_button("ask", "Ask"),
        class_="ask-bar",
    ),

    ui.output_ui("results"),
)


def server(input, output, session):
    """Run the Shiny server for the Amazon Beauty Shopping Assistant.

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
    hybrid = load_hybrid_retriever()
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
    def results():
        """Render the assistant's answer and its supporting reviews.

        Returns
        -------
        Any
            Shiny UI output.
        """
        if input.ask() == 0:
            return ui.p(
                "Ask about a product, ingredient, or use case -- for example "
                "“is this good for sensitive skin?”",
                class_="empty-state",
            )

        query = input.query().strip()
        if not query:
            return ui.p("Type a question to get started.", class_="empty-state")

        answer, results_docs, error = ask_assistant(
            query=query,
            hybrid=hybrid,
            llm_pipeline=llm_pipeline,
        )

        if error is not None:
            return ui.div(error, class_="error-banner")

        answer_message = ui.div(
            ui.div("🛍️", class_="assistant-avatar"),
            ui.p(answer or "I couldn't find a confident answer for that.", class_="assistant-text"),
            class_="assistant-message",
        )

        cards = render_review_cards(
            results_docs=results_docs,
            query=query,
            input=input,
            feedback_file=feedback_file,
            prev_counts=prev_counts,
        )

        if not cards:
            return answer_message

        return ui.TagList(
            answer_message,
            ui.div("What customers are saying", class_="evidence-heading"),
            *cards,
        )


app = App(app_ui, server)
