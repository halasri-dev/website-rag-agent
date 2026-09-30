"""Run the evaluation set. Usage: python eval/run_eval.py
Optional: set EVAL_DELAY (seconds between questions) for models with low per-minute limits."""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")
from graph import ask, MODEL_NAME  # noqa: E402

QUESTIONS = Path("eval/questions.json")
RESULTS = Path(f"eval/results_{MODEL_NAME}.json")
DELAY = float(os.environ.get("EVAL_DELAY", "2"))


def passed(q, r):
    refused = not r["enough_info"]
    answer = r["answer"].lower()
    kw_hits = sum(1 for k in q["must_include"] if k.lower() in answer)
    keyword_hit = (not q["must_include"]) or kw_hits >= q.get("min_keywords", 1)
    url_hits = sum(1 for u in q["expected_urls"] if u in r["sources"])
    url_ok = (not q["expected_urls"]) or url_hits >= q.get("min_url_matches", 1)

    if q["type"] == "unanswerable":
        return refused
    if q["type"] == "misleading":
        return refused or (keyword_hit and url_ok and bool(q["must_include"]))
    return (not refused) and keyword_hit and url_ok


def main():
    print(f"Model: {MODEL_NAME}  |  delay between questions: {DELAY}s\n")
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    rows = []
    for q in questions:
        try:
            r = ask(q["question"])
        except RuntimeError as e:  # daily quota used up
            print(f"\nStopped at question {q['id']}: {e}")
            break
        ok = passed(q, r)
        rows.append({**q, "answer": r["answer"], "sources": r["sources"], "evidence": r.get("evidence", ""),
                     "refused": not r["enough_info"], "best_distance": round(r["best_distance"], 3),
                     "passed": ok, "input_tokens": r["input_tokens"], "output_tokens": r["output_tokens"]})
        print(f'#{q["id"]:2d} [{q["type"]:15s}] {"PASS" if ok else "FAIL"}  {q["question"]}')
        print(f'     -> {r["answer"][:220]}')
        print(f'     sources: {r["sources"]}')
        time.sleep(DELAY)

    if not rows:
        print("No questions completed, nothing saved.")
        return
    RESULTS.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Summary ===")
    print(f"Completed {len(rows)}/{len(questions)} questions")
    for t in dict.fromkeys(r["type"] for r in rows):
        group = [r for r in rows if r["type"] == t]
        print(f'{t:15s} {sum(r["passed"] for r in group)}/{len(group)}')
    print(f'TOTAL           {sum(r["passed"] for r in rows)}/{len(rows)}')
    print(f'Tokens: input={sum(r["input_tokens"] for r in rows):,}  output={sum(r["output_tokens"] for r in rows):,}')
    print(f"Saved to {RESULTS}")


if __name__ == "__main__":
    main()