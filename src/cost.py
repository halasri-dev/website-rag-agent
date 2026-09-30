"""Cost analysis from evaluation results.  Usage: python src/cost.py"""
import json
from pathlib import Path

# USD per 1M tokens (input, output), paid tier, checked 30 Sep 2026.
# gemini-3.8-flash is an introductory price through 31 Dec 2026 (doubles on 1 Jan 2027).
PRICES = {
    "gemini-3.8-flash": (0.75, 3.75),
    "gemini-3.6-flash": (1.50, 7.50),
    "gemini-3.5-flash-lite": (0.30, 2.50),
}
HOSTED_EMBEDDING_PRICE = 0.15   # USD per 1M tokens, gemini-embedding-001 (comparison only)
VOLUMES = [100, 1_000, 10_000]
EXAMPLE_ID = 1   # which evaluation question to show as the worked example
OUT = Path("docs/cost_analysis.md")


def usd(inp, out, price):
    return (inp * price[0] + out * price[1]) / 1_000_000


def ingestion_tokens():
    log = Path("logs/usage.jsonl")
    if not log.exists():
        return None
    rows = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines()]
    ing = [r for r in rows if r["kind"] == "ingestion"]
    return ing[-1]["input_tokens"] if ing else None


def main():
    md = ["# Cost analysis", "",
          "Token counts are measured (from the API's usage metadata during the evaluation run). "
          "Prices are Google's published paid-tier rates; the free tier was used for development, "
          "so actual spend was $0. Output tokens include the model's thinking tokens.", ""]

    ing = ingestion_tokens()
    md += ["## Ingestion (one-time)", "",
           "- Crawling uses plain HTTP requests (free). No LLM calls happen during ingestion."]
    if ing:
        md += [f"- Chunks are embedded **locally** with `BAAI/bge-small-en-v1.5`: {ing:,} tokens, **$0**.",
               f"- For comparison, the same corpus embedded with a hosted model (gemini-embedding-001, "
               f"${HOSTED_EMBEDDING_PRICE:.2f} per 1M tokens) would cost about "
               f"**${ing * HOSTED_EMBEDDING_PRICE / 1_000_000:.3f}**. Token counts differ slightly between tokenizers."]
    else:
        md += ["- Chunks are embedded locally (`BAAI/bge-small-en-v1.5`): **$0**."]
    md += ["- Each query also embeds the question locally ($0).", ""]

    for f in sorted(Path("eval").glob("results_*.json")):
        model = f.stem.replace("results_", "")
        price = PRICES.get(model)
        if not price:
            continue
        rows = json.loads(f.read_text(encoding="utf-8"))
        llm_rows = [r for r in rows if r["input_tokens"] > 0]
        avg_in = sum(r["input_tokens"] for r in llm_rows) / len(llm_rows)
        avg_out = sum(r["output_tokens"] for r in llm_rows) / len(llm_rows)
        per_q = usd(avg_in, avg_out, price)

        md += [f"## Queries: {model}", "",
               f"Price: ${price[0]:.2f} / 1M input tokens, ${price[1]:.2f} / 1M output tokens.", ""]
        ex = next((r for r in rows if r["id"] == EXAMPLE_ID), None)
        if ex and ex["input_tokens"] > 0:
            md += [f"**Example query:** \"{ex['question']}\"", "",
                   f"- Input tokens: {ex['input_tokens']:,} (system prompt + 8 retrieved chunks + question)",
                   f"- Output tokens: {ex['output_tokens']:,}",
                   f"- Cost: **${usd(ex['input_tokens'], ex['output_tokens'], price):.5f}**", ""]
        md += [f"**Average over {len(llm_rows)} answered evaluation questions:** "
               f"{avg_in:,.0f} input + {avg_out:,.0f} output tokens = **${per_q:.5f} per query**.", "",
               "| Queries | Estimated cost |", "|---|---|"]
        md += [f"| {v:,} | ${per_q * v:,.2f} |" for v in VOLUMES]
        md += ["",
               "Estimate assumes every query reaches the LLM. Off-topic questions are refused by the "
               "retrieval distance check before any LLM call, so they cost $0.", ""]

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()