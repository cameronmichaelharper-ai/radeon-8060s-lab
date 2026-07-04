"""07_read_page.py -- downstream "read the retrieved page" step via Lemonade vision.

Demonstrates the second half of the vision-native pipeline: given a page image
(as would be returned by ColPali retrieval), a vision-capable LLM reads it and
extracts the answer. Uses Lemonade's OpenAI-compatible chat/completions with the
already-local vision Gemma (mmproj).

Usage:
  python 07_read_page.py <png_path> "<question>"
"""
import sys
import base64
import json
import urllib.request

LEMONADE = "http://localhost:13305/api/v1/chat/completions"
MODEL = "Gemma-4-26B-A4B-it-GGUF"  # vision-labeled, mmproj already downloaded


def read_page(png_path, question):
    with open(png_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    payload = {
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text":
                    "You are reading one page of a scientific paper. Answer ONLY from what is "
                    "visible on this page. Quote the exact numbers/values. If the answer is not "
                    "on this page, say 'NOT ON THIS PAGE'.\n\nQuestion: " + question},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ],
        }],
        "temperature": 0.0,
        "max_tokens": 3000,  # Gemma-4 is a reasoning model; leave room for reasoning + answer
    }
    req = urllib.request.Request(
        LEMONADE, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        resp = json.load(r)
    msg = resp["choices"][0]["message"]
    content = (msg.get("content") or "").strip()
    if not content:  # ran out before final answer, or model kept it in reasoning
        rc = (msg.get("reasoning_content") or "").strip()
        content = "[from reasoning_content]\n" + rc if rc else "[empty response]"
    return content


if __name__ == "__main__":
    png, q = sys.argv[1], sys.argv[2]
    print(f"PAGE: {png}\nQ: {q}\n---")
    print(read_page(png, q))
