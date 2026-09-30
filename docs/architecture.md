# Architecture

```mermaid
flowchart TB
  subgraph ING["Ingestion (run once)"]
    A["litestream.io sitemap"] --> B["crawl.py: robots.txt check, clean text"]
    B --> C[("data/pages.jsonl")]
    C --> D["ingest.py: drop short pages, chunk 1000 / 150 overlap"]
    D --> E["Local embeddings: bge-small-en-v1.5"]
    E --> F[("Chroma vector DB")]
  end
  subgraph QRY["Query: LangGraph workflow"]
    Q["Question"] --> R["retrieve: sub-queries, top 8 chunks, max 3 per page"]
    R --> G{"best distance under 0.9?"}
    G -- "no" --> X["refuse: no LLM call"]
    G -- "yes" --> L["generate: Gemini returns JSON with evidence"]
    L --> V["verify: citations come from chunk metadata"]
    V --> O["Answer + source URLs, or not enough information"]
    X --> O
  end
  F -.-> R
  L --> U["usage.jsonl: token log"]
```

## Components

| Component | File | Role |
|---|---|---|
| Crawler | `src/crawl.py` | Reads the sitemap, checks `robots.txt`, extracts clean text |
| Ingestion | `src/ingest.py` | Filters short pages, chunks, embeds locally, stores in Chroma |
| Workflow | `src/graph.py` | LangGraph: retrieve, route, generate or refuse |
| CLI | `src/cli.py` | Ask questions, show token totals |
| Usage log | `src/usage.py` | One JSON line per ingestion run or model call |
| Cost | `src/cost.py` | Turns measured tokens into cost estimates |

The same diagram is saved as an image in `docs/architecture.png`.