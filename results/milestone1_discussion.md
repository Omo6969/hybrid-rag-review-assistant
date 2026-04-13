# Milestone 1 - Qualitative Evaluation 

This document contains the comparison of BM25 and Semantic retrieval for selected queries and a short summary of insights.

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
- Both methods perform well since this is a simple keyword query.
- BM25 returns highly keyword-matching results.
- Semantic search returns similar results but slightly more natural phrasing.
- **Better Method:** Tie (slight edge to BM25 for precision)


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
- BM25 performs better by retrieving more specific and descriptive results.
- Semantic search results are repetitive and less informative.
- **Better Method:** BM25  
- **Reason:** Better diversity and specificity.


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
- BM25 retrieves more context-specific results (mentions of face).
- Semantic search includes some generic or vague results.
- **Better Method:** BM25  
- **Reason:** Better alignment with query intent (“for face”).


### Query 4: "something for dry skin"

**BM25 Top-5 Results:**
1. Convienent  
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
- BM25 performs poorly; many results are irrelevant or misleading.
- Semantic search captures the intent (“dry skin”) better.
- However, semantic results are still quite generic.
- **Better Method:** Semantic Search  
- **Reason:** Better intent matching, though still weak overall.


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
- BM25 retrieves mostly relevant results, though one is unrelated (“pet fur”).
- Semantic search has missing/weak results and lacks specificity.
- **Better Method:** BM25  
- **Reason:** More consistent relevance despite minor noise.


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