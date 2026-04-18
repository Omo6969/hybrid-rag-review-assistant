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

---

## Step 2 - Semantic RAG Pipeline

### 2.1 Retrieval

The semantic retriever reuses the `sentence-transformers/all-MiniLM-L6-v2` model from Milestone 1, now wrapped as a LangChain FAISS vectorstore via `build_semantic_vectorstore()` in `src/rag_pipeline.py`. The vectorstore is converted to a LangChain retriever with `k=5` (top-5 documents per query).

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

The full pipeline is assembled with `build_rag_chain()` using LangChain LCEL pipes:

```
retriever | build_context -> prompt_template -> llm -> StrOutputParser
```

Both semantic and hybrid retrievers are plug-compatible with this chain.

---

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

The hybrid approach is most beneficial for product queries that mix specific keywords with natural language intent, which is typical of real shopping queries.
