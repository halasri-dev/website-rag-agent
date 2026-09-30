# Cost analysis

Token counts are measured (from the API's usage metadata during the evaluation run). Prices are Google's published paid-tier rates; the free tier was used for development, so actual spend was $0. Output tokens include the model's thinking tokens.

## Ingestion (one-time)

- Crawling uses plain HTTP requests (free). No LLM calls happen during ingestion.
- Chunks are embedded **locally** with `BAAI/bge-small-en-v1.5`: 141,810 tokens, **$0**.
- For comparison, the same corpus embedded with a hosted model (gemini-embedding-001, $0.15 per 1M tokens) would cost about **$0.021**. Token counts differ slightly between tokenizers.
- Each query also embeds the question locally ($0).

## Queries: gemini-3.5-flash-lite

Price: $0.30 / 1M input tokens, $2.50 / 1M output tokens.

**Example query:** "What command restores a database from a Litestream replica?"

- Input tokens: 2,488 (system prompt + 8 retrieved chunks + question)
- Output tokens: 66
- Cost: **$0.00091**

**Average over 14 answered evaluation questions:** 2,383 input + 78 output tokens = **$0.00091 per query**.

| Queries | Estimated cost |
|---|---|
| 100 | $0.09 |
| 1,000 | $0.91 |
| 10,000 | $9.11 |

Estimate assumes every query reaches the LLM. Off-topic questions are refused by the retrieval distance check before any LLM call, so they cost $0.
