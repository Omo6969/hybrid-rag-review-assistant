# Milestone 2 - Discussion

## LLM Pipeline

### 1.1 Model Choice

**Model:** `llama-3.1-8b-instant` via [Groq API](https://console.groq.com)  
**Interface:** `langchain_groq.ChatGroq`

### 1.2 Rationale

| Factor | Decision |
|--------|----------|
| **Model family** | Llama 3 (Meta) - strong instruction-following, widely benchmarked |
| **Parameter count** | 8B - good balance of quality and speed |
| **Inference provider** | Groq - free tier, low latency (~300 tokens/s), no local GPU required |
| **Context window** | 8,192 tokens - sufficient for top-5 retrieved review chunks |
| **Tool/chat support** | Yes - supports system/human message structure needed for RAG |

Llama 3 8B was chosen over larger alternatives (e.g., Mistral 7B, Llama 3 70B) because:

1. **No local GPU required** - Groq hosts the model, eliminating hardware constraints on a standard laptop.
2. **Free tier availability** - Groq's free tier provides sufficient rate limits for development and evaluation without cost.
3. **Quality at 8B scale** - Llama 3 8B outperforms many 7B-class models on instruction-following tasks, making it a strong baseline for RAG.
4. **LangChain compatibility** - `langchain_groq.ChatGroq` integrates cleanly with LangChain prompt templates, making it straightforward to extend the pipeline to full RAG in later steps.

### 1.3 Implementation

The LLM pipeline is implemented in `src/rag_pipeline.py` as the `LLMPipeline` class with two operating modes:

- **Plain generation** - answers directly from model knowledge, no retrieval context
- **RAG generation** - answers grounded in retrieved review documents (enabled when `documents` are passed to `generate()`)

The class uses two separate `ChatPromptTemplate` objects to enforce different system prompts for each mode, ensuring the model correctly attributes answers to retrieved context when available.

For Milestone 2 notebook experiments, the saved semantic and hybrid retrievers were used directly where possible to avoid rebuilding retrieval artifacts unnecessarily.

## Step 2 - Semantic RAG Pipeline

### 2.1 Retrieval

For semantic RAG experiments, the project reuses the saved `SemanticRetriever` from Milestone 1 rather than rebuilding the semantic index inside the notebook. This choice improves efficiency, reduces notebook memory overhead, and keeps the experiments aligned with the same persisted retrieval artifacts used elsewhere in the project.

The semantic retriever uses the `sentence-transformers/all-MiniLM-L6-v2` model and a FAISS index built over the cleaned All Beauty retrieval documents. For each query, the retriever returns the top-`k` most semantically similar review documents, with `k=5` used in the Milestone 2 experiments.

### 2.2 Context Building

The `build_context()` function formats each retrieved `Document` as a numbered block containing:
- Product ASIN
- Review title
- Star rating
- Up to 400 characters of review text

This structure gives the LLM enough product identity context to ground its answer and cite specific products when using the `"detailed"` prompt variant.

### 2.3 Prompt Variants

Three system prompt variants were evaluated on the same test queries:

| Variant | Description | Observed behaviour |
|---------|-------------|-------------------|
| `minimal` | Single-sentence instruction, very short | Tends to produce one-line answers; sometimes drops nuance from reviews |
| `concise` | Shopping assistant with grounding constraint | Best balance - clear, factual, stays on topic, admits gaps |
| `detailed` | Encourages ASIN citation and pros/cons | Longest answers; useful when reviews disagree; can be verbose |

**Chosen default:** `concise` - it reliably grounds answers in the retrieved context without being overly restrictive or verbose.

### 2.4 RAG Pipeline

The semantic RAG workflow was implemented as a lightweight custom pipeline for notebook experimentation. The process follows the standard RAG structure:

1. retrieve top-`k` supporting documents with the semantic retriever,
2. build a structured context block from the retrieved reviews,
3. apply a selected prompt template, and
4. generate a grounded answer with the LLM.

This approach satisfies the milestone requirement for a retrieval -> context -> prompt -> LLM workflow while remaining more efficient and stable than rebuilding a full LangChain retriever pipeline inside the notebook.

## Step 3 - Hybrid RAG Pipeline

### 3.1 Hybrid Retriever Design

`src/hybrid.py` implements `HybridRetriever`, which combines BM25 and semantic search using **Reciprocal Rank Fusion (RRF)**:

```
RRF(d) = sum_r  1 / (k + rank_r(d))    where k = 60
```

RRF was chosen over simple score averaging because:
- BM25 and semantic scores are on different scales (not directly comparable)
- RRF only depends on rank position, making it scale-invariant
- Documents ranked highly by both retrievers receive the highest combined score

### 3.2 Hybrid vs Semantic RAG

| Query type | Semantic RAG | Hybrid RAG |
|------------|-------------|------------|
| Keyword-heavy ("lip balm SPF") | Moderate - misses exact terms | Better - BM25 catches keyword matches |
| Intent-based ("something for dry skin") | Good - understands meaning | Similar - semantic component carries it |
| Mixed ("fragrance-free moisturizer for sensitive skin") | Good | Best - combines both signals |

In general, hybrid RAG was most useful for queries that combined explicit keywords with broader intent. BM25 helped recover exact term matches, while semantic retrieval improved coverage for more descriptive phrasing. As a result, hybrid retrieval often provided stronger supporting evidence for generation than semantic retrieval alone, especially on mixed-intent shopping queries.

## Step 4: Qualitative Evaluation Results

The Hybrid RAG workflow was manually evaluated on 5 queries from the Milestone 1 query set using the three milestone criteria:

- **Accuracy** - whether the answer was factually correct based on the retrieved reviews
- **Completeness** - whether the answer addressed all important aspects of the query
- **Fluency** - whether the answer was natural, clear, and easy to read

### 4.1 Evaluation Table

| Query ID | Query | Difficulty | Accuracy | Completeness | Fluency | Notes |
|---|---|---|---|---|---|---|
| 1 | lip balm | easy | Yes | Yes | Yes | The answer was relevant, grounded in the retrieved reviews, and clearly written. It identified the product correctly and summarized review evidence well. |
| 2 | face moisturizer | easy | Yes | Yes | Yes | The answer correctly identified the product type and presented the supporting review evidence in a concise and readable way. |
| 3 | sunscreen for face | easy | Yes | Yes | Yes | The answer remained grounded in the retrieved reviews and addressed the query directly. It was both accurate and easy to understand. |
| 4 | something for dry skin | medium | Yes | No | Yes | The answer was plausible and grounded, but the query was broader and more open-ended. The response did not fully explore multiple possible product options, so completeness was weaker. |
| 5 | product to reduce frizzy hair | medium | Yes | No | Yes | The answer was relevant and fluent, and it reflected the retrieved review evidence. However, it did not compare alternative products or discuss trade-offs in much detail. |

### 4.2 Summary of Key Observations

Overall, the Hybrid RAG workflow performed well on straightforward product-search queries and produced answers that were generally accurate, fluent, and grounded in the retrieved review evidence. The strongest results were observed on easy queries such as *lip balm*, *face moisturizer*, and *sunscreen for face*, where the retrieval task was relatively direct and the generated answers remained concise and relevant.

Performance was slightly weaker on broader or more descriptive queries such as *something for dry skin* and *product to reduce frizzy hair*. In these cases, the generated answers were still mostly accurate and fluent, but completeness was lower because the queries allowed multiple interpretations and the system did not always compare a range of candidate products in detail.

### 4.3 Overall Reflection

The Hybrid RAG workflow is performing reasonably well for Milestone 2. It successfully combines retrieval and generation so that answers are grounded in the review corpus rather than being produced without evidence. In practice, hybrid retrieval appears more robust than using a single retrieval method, since BM25 contributes exact keyword matching while semantic retrieval helps recover relevant documents even when the wording differs.

### 4.4 Limitations

1. The generated answers depend heavily on the quality and coverage of the retrieved reviews.  
2. Broader or underspecified queries reduce completeness because multiple interpretations are possible.  
3. The cleaned dataset contains limited metadata, so answers rely mainly on review text rather than richer product attributes.  
4. The workflow does not always compare multiple candidate products in depth.

### 4.5 Possible Improvements

1. Improve hybrid reranking and experiment further with retrieval weights.  
2. Add richer metadata into the context when available.  
3. Tune retrieval depth more systematically to balance coverage and focus.  
4. Add clearer source attribution in generated answers to strengthen transparency and grounding.
