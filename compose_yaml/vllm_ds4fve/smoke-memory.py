#!/usr/bin/env python3
"""Send one bounded text or native image request and persist the response."""
import argparse
import base64
import json
import struct
import time
import urllib.request
import zlib
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("phase", choices=["text", "image"])
parser.add_argument("--url", default="http://127.0.0.1:8888/v1/chat/completions")
parser.add_argument("--output", required=True)
args = parser.parse_args()
if args.phase == "text":
    content = "대한민국의 수도를 한 문장으로 답해 줘."
else:
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)
    width, height = 512, 256
    row = b"\x00" + bytes([255, 0, 0]) * (width // 2) + bytes([0, 0, 255]) * (width // 2)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(row * height)) + chunk(b"IEND", b"")
    content = [{"type": "text", "text": "이미지의 왼쪽과 오른쪽 색상을 각각 말해 줘."}, {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}}]
payload = {"model": "deepseek-v4-flash-vision-exp", "messages": [{"role": "user", "content": content}], "max_tokens": 512, "temperature": 0, "chat_template_kwargs": {"thinking": False}}
request = urllib.request.Request(args.url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
started = time.monotonic()
with urllib.request.urlopen(request, timeout=300) as response:
    result = json.load(response)
output = {"phase": args.phase, "elapsed_seconds": time.monotonic() - started, "response": result}
Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(output, ensure_ascii=False, indent=2))
if not result.get("choices") or not result["choices"][0].get("message", {}).get("content"):
    raise SystemExit("No terminal text content in response")
