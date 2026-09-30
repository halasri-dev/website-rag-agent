"""Usage:
  python src/cli.py "your question"     ask one question
  python src/cli.py                     interactive mode
  python src/cli.py --usage             show token totals
"""
import json
import sys
from pathlib import Path


def show_usage():
    log = Path("logs/usage.jsonl")
    if not log.exists():
        print("No usage logged yet.")
        return
    totals = {}
    for line in log.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        t = totals.setdefault(r["kind"], {"calls": 0, "in": 0, "out": 0})
        t["calls"] += 1
        t["in"] += r["input_tokens"]
        t["out"] += r["output_tokens"]
    for kind, t in totals.items():
        print(f'{kind:10s} calls={t["calls"]:4d}  input_tokens={t["in"]:,}  output_tokens={t["out"]:,}')


def show(result):
    print("\nAnswer:", result["answer"])
    if result["sources"]:
        print("Sources:")
        for u in result["sources"]:
            print("  -", u)
    print(f'(best distance {result["best_distance"]:.3f} | '
          f'tokens in={result["input_tokens"]} out={result["output_tokens"]})\n')


def main():
    args = sys.argv[1:]
    if args == ["--usage"]:
        show_usage()
        return
    from graph import ask  # imported here so --usage stays fast
    if args:
        show(ask(" ".join(args)))
        return
    print("Ask about Litestream (empty line to quit)")
    while True:
        q = input("> ").strip()
        if not q:
            break
        show(ask(q))


if __name__ == "__main__":
    main()