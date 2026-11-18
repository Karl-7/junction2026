"""List the exact Gemini model ids your API key can call with generateContent.

    python -m llm_bench.gemini_models            # all
    python -m llm_bench.gemini_models flash      # filter
Use the printed id (without the "models/" prefix) in llm_bench/models.json.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request

from .config import load_env


def main():
    load_env()
    key = os.environ.get("GEMINI_API_KEY") or sys.exit("GEMINI_API_KEY is not set (put it in .env)")
    base = os.environ.get("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
    flt = (sys.argv[1] if len(sys.argv) > 1 else "").lower()
    token, rows = "", []
    while True:
        req = urllib.request.Request(f"{base}/models?pageSize=200" + (f"&pageToken={token}" if token else ""),
                                     headers={"x-goog-api-key": key})
        with urllib.request.urlopen(req, timeout=60) as r:
            d = json.loads(r.read())
        for m in d.get("models", []):
            if "generateContent" in m.get("supportedGenerationMethods", []):
                rows.append((m["name"].removeprefix("models/"), m.get("displayName", "")))
        token = d.get("nextPageToken")
        if not token:
            break
    for mid, name in sorted(rows):
        if flt in mid.lower() or flt in name.lower():
            print(f"{mid:45} {name}")


if __name__ == "__main__":
    main()
