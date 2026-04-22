# Final Discussion

## Step 1: Improve Your Workflow

### Dataset Scaling

The cleaned All Beauty dataset contains **701,092 records**, which already exceeds the final submission minimum requirement of **10,000 products**. Because the project had already been scaled to this larger corpus, no additional sampling expansion was needed for the final submission.

### LLM Experiment

- Models compared (name, family, size)
- Results and discussions
    - Prompt used (copy it here)
    - Results
- Which model you chose and why

## Step 2: Additional Feature (Option 1: Quantitative Evaluation)

### What You Implemented

Quantitative evaluation was added using custom retrieval metrics rather than `RAGAS`. The evaluation focused on **precision@5** and **recall@5**, which are explicitly suggested in the final submission brief. A small manually labeled relevance set was created for five representative queries, and BM25, semantic retrieval, and hybrid retrieval were compared on the same query set. :contentReference[oaicite:1]{index=1}

- **precision@5** measures the proportion of relevant documents among the top 5 retrieved results.
- **recall@5** measures the proportion of all relevant documents that appear in the top 5 retrieved results.

This approach was chosen because it is lightweight, transparent, and easy to reproduce within the available project scope.

### Key results

| Retriever | Precision@5 | Recall@5 |
|---|---:|---:|
| Hybrid | 0.80 | 0.5524 |
| Semantic | 0.80 | 0.5488 |
| BM25 | 0.76 | 0.5298 |

### Interpretation

The quantitative evaluation shows that **Hybrid retrieval** achieved the strongest overall performance, with the highest average precision@5 and recall@5 across the evaluation queries. However, the margin over **Semantic retrieval** was small, which suggests that semantic retrieval alone was already quite strong on this query set.

**BM25** remained competitive, especially for more direct product-oriented queries, but performed slightly worse overall than the other two methods. This pattern is consistent with the system design: BM25 benefits from exact lexical overlap, semantic retrieval captures broader query meaning, and hybrid retrieval combines both signals. As a result, hybrid retrieval appears to offer a modest but consistent improvement over using only one retrieval method.
  
## Step 3: Improve Documentation and Code Quality

### Documentation Update

- Summary of `README` improvements

### Code Quality Changes

- Summary of cleanups

## Step 4: Cloud Deployment Plan

(See Step 4 above for required subsections)