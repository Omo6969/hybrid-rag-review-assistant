# Milestone 1 - Qualitative Evaluation

This document presents a qualitative comparison of BM25 and semantic retrieval on selected queries, followed by a summary of the main patterns observed across both methods.

## Full Query Set Used

The qualitative evaluation used a query set of 10 examples spanning three difficulty levels: easy, medium, and complex.

| Query ID | Query                                                | Difficulty |
| -------- | ---------------------------------------------------- | ---------- |
| 1        | lip balm                                             | easy       |
| 2        | face moisturizer                                     | easy       |
| 3        | sunscreen for face                                   | easy       |
| 4        | something for dry skin                               | medium     |
| 5        | product to reduce frizzy hair                        | medium     |
| 6        | gentle makeup remover                                | medium     |
| 7        | beauty product that is easy to carry while traveling | complex    |
| 8        | makeup remover that does not irritate sensitive skin | complex    |
| 9        | skin care product for very dry lips in winter        | complex    |
| 10       | lightweight product that keeps skin hydrated all day | complex    |

These queries were designed to include direct keyword searches, moderately descriptive natural-language searches, and more complex intent-based searches so that BM25 and semantic retrieval could be compared across different query types.

## 4.3 Comparison of Retrieval Methods

### Query 1: "lip balm"

**BM25 Top-5 Results:**

1. best lip balm  
2. Best lip balm ever  
3. beautiful lip balm  
4. Great daily lip balm!  
5. PRETTY LIP BALM

**Semantic Search Top-5 Results:**

1. Lip balm  
2. The best lip balm to exist.  
3. Great lip balm  
4. My fav lip balm  
5. Very good lip balm. Too expensive.  

**Comparison & Comments:**

- Both methods perform well because this is a simple keyword-based query and the phrase *lip balm* appears directly in many retrieved results.
- BM25 has a slight advantage in precision because its top results align very closely with the exact query terms.
- Semantic search also performs strongly, but its results are not meaningfully better in usefulness since they remain very similar in wording and intent to the BM25 results.
- **Better Method:** Tie (slight edge to BM25 for precision)
- **Reason:** Both methods retrieve highly relevant results, but BM25 benefits slightly from exact keyword matching on this straightforward query.

### Query 2: "face moisturizer"

**BM25 Top-5 Results:**

1. Nice lightweight face moisturizer  
2. Great moisturizer!  
3. Great Face Moisturizer!  
4. Good product  
5. Moisturizer  

**Semantic Search Top-5 Results:**

1. Moisturizer  
2. Good product  
3. nice moisturizer  
4. Moisturizer  
5. Moisturizer  

**Comparison & Comments:**

- BM25 performs better for this query because several of its top results explicitly mention both *face* and *moisturizer*, making them more specific and more useful to the user’s intent.
- In contrast, the semantic search results are repetitive and generic, with multiple entries such as *Moisturizer* and *Good product* that provide little detail or distinction.
- Although both methods retrieve items related to moisturizer, BM25 gives a clearer sense that the results are actually about a face moisturizer rather than a generic skincare product.
- **Better Method:** BM25
- **Reason:** Better specificity, diversity, and usefulness for the query intent.

### Query 3: "sunscreen for face"

**BM25 Top-5 Results:**

1. Sunscreen no. Face paint, meh.  
2. Summer Sunscreen application  
3. Great face sunscreen  
4. Great sunscreen  
5. Love this sunscreen for my face  

**Semantic Search Top-5 Results:**

1. Great sunscreen for face  
2. Great Product  
3. sunscreen  
4. Wonderful sunscreen  
5. Recommended  

**Comparison & Comments:**

- BM25 performs better for this query because several of its top results explicitly mention both *sunscreen* and *face*, which makes them more closely aligned with the user’s intent.
- Semantic search retrieves one highly relevant result, but the remaining results are more generic, including entries such as *Great Product*, *sunscreen*, and *Recommended*, which provide little context.
- Although BM25 also includes one weaker result, its overall ranking is more useful because it returns more face-specific items across the top results.
- **Better Method:** BM25
- **Reason:** Better alignment with the query intent and more context-specific results.

### Query 4: "something for dry skin"

**BM25 Top-5 Results:**

1. Convenient  
2. Five Stars  
3. Dries skin out  
4. My dry skin itches like crazy, so I'm trying something new by exfoliating ...  
5. I have trouble exfoliating dry skin and these work pretty well! I wish they made something with a ...  

**Semantic Search Top-5 Results:**

1. Good  
2. To dry  
3. A must have  
4. Good for dry skin.  
5. Good for dry skin.  

**Comparison & Comments:**

- BM25 performs poorly for this query because several of its top results are irrelevant, overly generic, or even misleading, such as *Convenient* and *Five Stars*, which do not clearly address dry skin.
- Semantic search captures the query intent better by retrieving results that more directly relate to dry skin, even though many of them are still vague or low in descriptive value.
- This query highlights a case where semantic retrieval is more useful than BM25 for a natural-language need, but neither method performs especially strongly because the returned results remain generic.
- **Better Method:** Semantic Search
- **Reason:** Better intent matching and slightly higher usefulness for a vague, natural-language query, although overall retrieval quality is still limited.

### Query 5: "product to reduce frizzy hair"

**BM25 Top-5 Results:**

1. Leaves hair softer and less frizzy  
2. Reduce pet fur everywhere  
3. Frizzy  
4. Great product! It really takes care of frizzy hair ...  
5. Great product to make frizzy puffy hair :/  

**Semantic Search Top-5 Results:**

1. The only product to tame frizzy hair  
2. good buy  
3. (missing result)  
4. Four Stars  
5. Great Product  

**Comparison & Comments:**

- BM25 performs better for this query because most of its top results relate directly to frizzy hair and reflect the user’s intended need for a smoothing or taming product.
- Although BM25 includes one clearly irrelevant result about *pet fur*, its overall ranking is still more useful and more specific than the semantic search output.
- Semantic search retrieves one strong result, but the remaining results are weak, generic, or incomplete, which makes them less helpful for identifying a relevant hair-care product.
- **Better Method:** BM25
- **Reason:** More consistent relevance, better specificity, and greater usefulness despite minor noise in one result.

## Observations

- BM25 works well for:
  - keyword-heavy queries
  - product-specific queries
- Semantic search works better for:
  - vague or natural language queries (e.g., “something for dry skin”)
- Semantic search sometimes:
  - produces generic or repetitive results
  - lacks diversity
- BM25 sometimes:
  - retrieves irrelevant results due to keyword overlap

## 4.4 Summary

### Strengths of BM25

- Strong performance on keyword-based queries
- Produces more diverse and descriptive results
- Works well when query terms directly match documents
- More consistent ranking quality in most cases

### Weaknesses of BM25

- Fails for vague or natural language queries (e.g., “something for dry skin”)
- Cannot understand intent beyond exact words
- Can retrieve irrelevant results due to keyword overlap

### Strengths of Semantic Search

- Better at understanding user intent
- Performs better on vague/descriptive queries
- Handles natural language better than BM25

### Weaknesses of Semantic Search

- Often produces generic or repetitive results
- Lacks diversity in retrieved documents
- Sometimes returns incomplete or missing results
- Less precise for keyword-heavy queries

### Challenging Query Types for Both Methods

- Vague queries with little context (e.g., “something for dry skin”)
- Queries requiring multiple conditions or constraints
- Queries where user intent is ambiguous

### Where Advanced Methods Can Help

#### 1. Reranking

- Can improve ordering of retrieved results
- Helps remove irrelevant or low-quality results

#### 2. Hybrid Search (BM25 + Semantic)

- Combines keyword precision with semantic understanding
- Likely to improve overall retrieval performance

#### 3. RAG (Retrieval-Augmented Generation)

- Can interpret user intent more deeply
- Useful for complex or conversational queries

### Conclusion

BM25 demonstrated stronger overall performance in this evaluation, largely due to the keyword-heavy nature of the dataset and queries. However, semantic search showed clear advantages in handling natural language and intent-based queries.

These results suggest that neither method alone is sufficient for all query types. A hybrid approach, potentially combined with reranking or RAG, would provide the most robust and effective retrieval system in real-world applications.
