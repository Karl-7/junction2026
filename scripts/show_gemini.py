"""Print a Gemini agent run (policies/gemini_agent.py JSONL) as a readable raw transcript.

    python -m scripts.show_gemini runs/gemini/seed101.jsonl [--no-thoughts] [--width 600]
"""
from __future__ import annotations

import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--no-thoughts", action="store_true")
    ap.add_argument("--width", type=int, default=0, help="truncate long lines (0 = no truncation)")
    a = ap.parse_args()
    cut = (lambda s: s if not a.width or len(s) <= a.width else s[:a.width] + " …")
    for line in open(a.path):
        r = json.loads(line)
        ev = r["event"]
        if ev == "response":
            cand = (r["raw"].get("candidates") or [{}])[0]
            for p in (cand.get("content") or {}).get("parts", []):
                if p.get("thought") and "text" in p:
                    if not a.no_thoughts:
                        print(cut(f"[{r['n']:>3}] THOUGHT: " + p["text"].strip().replace("\n", "\n            ")))
                elif "text" in p:
                    print(cut(f"[{r['n']:>3}] SAYS: " + p["text"].strip()))
                elif "functionCall" in p:
                    fc = p["functionCall"]
                    print(f"[{r['n']:>3}] CALL: {fc.get('name')}({json.dumps(fc.get('args', {}), ensure_ascii=False)})")
            if cand.get("finishReason") not in (None, "STOP"):
                print(f"[{r['n']:>3}] finishReason={cand.get('finishReason')}")
        elif ev == "tool":
            print(cut("      -> " + json.dumps(r["result"], ensure_ascii=False, separators=(",", ":"))))
        elif ev in ("http_error", "net_error"):
            print(f"      !! {ev} {r.get('code', '')} (retry {r['attempt']})")
        elif ev == "end":
            print("=== END, submitted:", json.dumps(r["submitted"], ensure_ascii=False))


if __name__ == "__main__":
    main()
