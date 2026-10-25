"""Check that the Gemini API key works for Gemma 4 with tool calling (one tiny request; never prints the key).

usage: python3 scripts/api_check.py [--key-file .gemini_key] [--model gemma-4-31b-it]
"""
import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://generativelanguage.googleapis.com/v1beta/openai"


def call(key, path, body=None):
    req = urllib.request.Request(f"{BASE}/{path}", method="POST" if body else "GET",
                                 data=json.dumps(body).encode() if body else None,
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key-file", type=Path, default=Path(__file__).resolve().parent.parent / ".gemini_key")
    ap.add_argument("--model", default="gemma-4-31b-it")
    a = ap.parse_args()
    key = a.key_file.expanduser().read_text().strip()
    try:
        models = [m["id"] for m in call(key, "models")["data"]]
        print("gemma models visible:", sorted(m for m in models if "gemma" in m))
    except urllib.error.HTTPError as e:
        print("model list failed:", e.code, e.read()[:200])
    tool = {"type": "function", "function": {"name": "read_file", "description": "Read a file.",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}
    t0 = time.time()
    try:
        d = call(key, "chat/completions", {"model": a.model, "tools": [tool], "messages": [
            {"role": "user", "content": "Use the read_file tool to open setup.py. Do not answer in text."}]})
        msg = d["choices"][0]["message"]
        print(f"{a.model}: tool_calls={json.dumps(msg.get('tool_calls'))[:200]} content={str(msg.get('content'))[:120]!r}")
        print("usage:", d.get("usage"), f"latency {time.time() - t0:.1f}s")
    except urllib.error.HTTPError as e:
        print(f"{a.model} call failed: HTTP {e.code} {e.read()[:300]!r}")


if __name__ == "__main__":
    main()
