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
