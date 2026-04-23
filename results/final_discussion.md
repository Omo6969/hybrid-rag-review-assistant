# Final Discussion

## Step 1: Improve Your Workflow

### Dataset Scaling

The cleaned All Beauty dataset contains **701,092 records**, which already exceeds the final submission minimum requirement of **10,000 products**. Because the project had already been scaled to this larger corpus, no additional sampling expansion was needed for the final submission.

### LLM Experiment

#### Models compared

| Model | Family | Parameters | Provider |
| ----- | ------ | ---------- | -------- |
| `llama-3.1-8b-instant` | Llama 3.1 (Meta) | 8B (baseline) | Groq |
| `llama-3.3-70b-versatile` | Llama 3.3 (Meta) | 70B (challenger) | Groq |

Both models were tested under identical conditions: same retriever (semantic, k=5), same prompt
template, temperature=0.0, max_tokens=512.

#### Prompt used

**System:**
> You are a helpful Amazon shopping assistant.
> Answer the question using ONLY the following product reviews.
> Be concise and factual. If the reviews do not answer the question, say so.

**Human:**
> Reviews:
> `{context}`
>
> Question: `{question}`

Retrieved context: top-5 documents from the semantic retriever, formatted as numbered blocks
containing ASIN, product title, star rating, and up to 400 characters of review text.

#### Results

Five queries spanning different difficulty levels were run through both models with identical
retrieved context.

**Query 1: Keyword-exact:** *What lip balm works best for extremely dry, chapped lips?*

**llama-3.1-8b-instant:**
> Based on the reviews, the following lip balms are recommended for extremely dry, chapped lips:
>
> - Best of Many Lip Balms (ASIN: B06XJNDV5T) — Reviewer states it lasts the longest and is very effective.
> - It's the best lip balm I've tried (ASIN: B07BY1M4H4) — Reviewer states it heals and makes lips soft.
> - Must have for dry lips (ASIN: B0011V3T8W) — Reviewer states it works better than any other lip balm.
> - My must have balm (ASIN: B01LQXI3H6) — Reviewer states it is the most moisturizing lip balm they have ever tried.
> - The only lip balm that works! (ASIN: B07KKN2G3X) — Reviewer states it is the only product that has really worked for their extremely dry lips.

**llama-3.3-70b-versatile:**
> According to reviews 1, 2, 3, 4, and 5, the respective lip balms work well for dry, chapped lips.
> Reviewers describe them as "the best", "heals", "must have", "most moisturizing", and "the only one
> that works". However, the specific product names are not mentioned in the reviews, only the ASIN numbers.

**Query 2: Vague intent:** *something for sensitive skin that won't break me out*

**llama-3.1-8b-instant:**
> Based on the reviews, the following products may be suitable for sensitive skin that won't break you out:
>
> - B07ZQS3P1J (Rating: 5/5) — "Sensitive skin great for sensitive skin"
> - B08RNQNFW1 (Rating: 5/5) — "Best Buy! suitable for sensitive skin."
> - B08L3J4FB9 (Rating: 5/5) — "I have sensitive skin and have trouble finding skin care that doesn't break me out! I absolutely love this product!"

**llama-3.3-70b-versatile:**
> Based on the reviews, the following products may be suitable for sensitive skin and won't break you out:
>
> - ASIN: B07ZQS3P1J (Rating: 5/5)
> - ASIN: B08RNQNFW1 (Rating: 5/5)
> - ASIN: B08L3J4FB9 (Rating: 5/5)

**Query 3: Multi-condition:** *fragrance-free moisturizer that also has SPF protection*

**llama-3.1-8b-instant:**
> Based on the reviews, there is no product that meets the criteria of being a fragrance-free
> moisturizer with SPF protection.

**llama-3.3-70b-versatile:**
> Review [1] mentions a product that is fragrance-free and has SPF protection, but the SPF value
> is not specified. Review [3] mentions a product with a natural SPF of 20, but it is not fragrance-free.
> Review [4] mentions a fragrance-free moisturizer, but it does not have SPF protection.
>
> Only review [1] meets the fragrance-free requirement and has SPF, but the SPF value is not specified.

**Query 4: Product comparison:** *Is Burt's Bees or EOS better for dry lips?*

**llama-3.1-8b-instant:**
> Based on the reviews, neither Burt's Bees nor EOS is specifically mentioned as being better for
> dry lips. Review 1 mentions Camax as the only brand that worked for the reviewer's sensitive lips,
> while Review 2 mentions Burt's Bees and Blistex as examples of pricey lip balms that left their
> lips dried out. Review 5 is about EOS, but it doesn't specifically mention dry lips.

**llama-3.3-70b-versatile:**
> The reviews do not provide a direct comparison between Burt's Bees and EOS for dry lips. However,
> review [4] mentions that the reviewer, a former Burt's Bees user, prefers the product being reviewed
> over Burt's Bees for dry lips. Review [2] also mentions that Burt's Bees dried out the reviewer's lips.
> Review [5] mentions that EOS keeps lips soft, but does not specifically address dry lips. Review [1]
> mentions that EOS did not work for the reviewer's sensitive lips. Overall, the reviews do not provide
> a clear answer to which one is better for dry lips.

**Query 5: Negative filter:** *hair serum that doesn't make hair greasy or weigh it down*

**llama-3.1-8b-instant:**
> Based on the reviews, the hair serum that doesn't make hair greasy or weigh it down is:
>
> - ASIN: B00XDB3QTA (reviewed 3 times)
> - ASIN: B000TFW8WE
> - ASIN: B08PPZLSYT

**llama-3.3-70b-versatile:**
> Reviews [1], [2], [3] and [5] mention that the hair serum does not make hair greasy or weigh it down.
> The ASINs for these products are B000TFW8WE, B00XDB3QTA, B08PPZLSYT.

#### Discussion

| Dimension | llama-3.1-8b-instant (8B) | llama-3.3-70b-versatile (70B) |
| --------- | ------------------------- | ----------------------------- |
| **Factual grounding** | Stays grounded; often gives direct product-style answers with ASINs and brief evidence | Stays grounded; tends to reference reviews more cautiously and synthesise across them |
| **Multi-condition queries** | Simpler handling; may stop at "no product found" without unpacking partial matches | Stronger; analyses each condition separately and identifies partial matches |
| **Product comparisons** | More likely to stop at the lack of a direct comparison | More likely to integrate evidence across multiple reviews and explain why the comparison remains inconclusive |
| **Simple shopping queries** | Often more direct and actionable | Still grounded, but sometimes less helpful because it is more cautious than necessary |
| **Negative-filter queries** | Clear and concise | Similar overall, with slightly more explicit multi-review synthesis |
| **Handling retrieval gaps** | Can be terse when evidence is incomplete | Better at explaining what is missing from the evidence |
| **Response style** | Direct, list-oriented, shopping-assistant style | More analytical, cautious, and synthesis-oriented |
| **Latency (Groq free tier)** | ~0.5–1 s | ~2–4 s |

The clearest advantage of the 70B model appears on queries that require reasoning across multiple retrieved documents, especially Query 3 and Query 4. In these cases, the 70B model synthesises evidence more carefully and is more transparent about partial matches and uncertainty.

For simpler shopping-style queries, the 8B model is sometimes more immediately useful because it produces direct and actionable answers. However, the 70B model is stronger overall when the query requires multi-document reasoning, handling ambiguity, or explaining why the retrieved evidence is incomplete.

#### Model chosen

**`llama-3.3-70b-versatile` is adopted as the new default.**

Although the 8B model performs well on straightforward queries and can be more direct in shopping-style responses, the 70B model demonstrates stronger reasoning across multiple retrieved reviews and is more transparent about gaps or ambiguity in the evidence. That makes it the better default for the final RAG pipeline, where grounded reasoning is more important than minimal latency.

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

The documentation was updated to make the repository clearer, more complete, and easier to reproduce. The `README.md` was revised to reflect the final project scope rather than only the earlier milestones. In particular, it now describes the retrieval and RAG workflow, the hybrid retriever, the updated application modes, the evaluation workflow, and the required environment setup. The README also includes clearer run instructions for the notebooks and app, updated repository structure, and guidance for using environment variables through `.env` and `.env.example`.

### Code Quality Changes

Several code quality and reproducibility improvements were made in the final submission. File handling was standardized using `pathlib.Path` rather than hardcoded paths where possible. API keys were kept out of source code and moved to environment variables. Function docstrings were added or improved across the codebase, including retrieval and RAG components. The project dependencies were updated to support the final RAG workflow and model experimentation. The notebook workflow was also simplified to reuse saved retrieval artifacts when available, which reduced unnecessary recomputation and improved stability during experimentation.

## Step 4: Cloud Deployment Plan

### Data Storage

For a cloud deployment on AWS, the raw Amazon review files would be stored in **Amazon S3** because they are large, static, and naturally suited to object storage. The processed datasets, including cleaned parquet and JSONL outputs, would also be stored in **S3**. The semantic vector index and BM25 index could likewise be stored in **S3** as versioned retrieval artifacts and loaded by the application at startup. This approach keeps storage simple, scalable, and decoupled from the application itself.

### Compute

The application could run on **AWS App Runner**, **ECS**, or a small **EC2** instance, depending on the desired level of operational complexity. For a lightweight deployment, App Runner would be attractive because it simplifies containerized web app deployment. If more control or scaling flexibility were needed, ECS would be a stronger option. Multiple users could be handled through horizontal scaling of application instances behind a load balancer. LLM inference would remain **API-based** rather than self-hosted, since using a hosted LLM API reduces the complexity and cost of running GPU-backed inference infrastructure.

### Streaming / Updates

New products or updated review files could be incorporated through a scheduled batch update pipeline. New raw data would first be uploaded to **S3**, then processed through a recurring job that rebuilds or refreshes the cleaned dataset and retrieval artifacts. A scheduler such as a cron job, GitHub Actions workflow, or a cloud-native scheduling service could trigger these updates automatically. This would allow the production system to stay current without requiring manual rebuilding of the full pipeline each time new data becomes available.