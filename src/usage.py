"""Append-only usage log: one JSON line per model call or ingestion run."""
import json
import time
from pathlib import Path

LOG = Path("logs/usage.jsonl")


def log_usage(kind, model, input_tokens, output_tokens, note=""):
    LOG.parent.mkdir(exist_ok=True)
    row = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "kind": kind,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "note": note,
    }
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")