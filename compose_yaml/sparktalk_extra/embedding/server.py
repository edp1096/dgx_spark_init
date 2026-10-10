"""Local CUDA-only text embedding worker. Model files must be prepared first."""
import json
import math
import os
import re
import threading
import time
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = "google/embeddinggemma-2"
REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
DIMENSIONS = 768
PROFILE = f"{MODEL}@{REVISION}:text:bf16:{DIMENSIONS}:chunk512-o48-v1"
lock = threading.Lock()
model = None
busy = False
quiescing = False
last_used = 0.0


def get_model():
    global model
    if model is None:
        import torch
        from sentence_transformers import SentenceTransformer
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA with native BF16 support is required")
        model = SentenceTransformer(os.environ.get("MODEL_DIR", "/models"), device="cuda",
            config_kwargs={"vision_config": None, "audio_config": None},
            model_kwargs={"torch_dtype": torch.bfloat16, "attn_implementation": "sdpa"},
            local_files_only=True)
        if sum(p.numel() for p in model.parameters()) > 300_000_000:
            model = None
            raise RuntimeError("unused modality encoders were loaded")
        if any(p.device.type != "cuda" for p in model.parameters()):
            model = None
            raise RuntimeError("CPU model offload is not permitted")
        model.max_seq_length = 8192
    return model


def literal_text(text):
    # Treat model-control/media markers in source code as literal text. Their
    # exact original spelling is retained in returned snippets and in SQLite.
    return re.sub(r"<\|([^\s|<>]+)\|>", lambda m: "< |" + m.group(1) + "| >", text)


def encode(inputs, task):
    m = get_model()
    out = []
    import numpy as np
    for index, item in enumerate(inputs):
        text = item["text"]
        if task == "query":
            tokens = m.tokenizer.encode(literal_text(text), add_special_tokens=False)
            if len(tokens) > 2048:
                raise ValueError("query exceeds 2048 tokens")
            vector = m.encode(literal_text(text), prompt_name="SearchQuery", normalize_embeddings=True,
                              convert_to_numpy=True, show_progress_bar=False)
            out.append({"input_index": index, "segment": 0, "text": text, "embedding": vector.tolist()})
            continue
        title = item.get("title", "").strip() or "none"
        tokenized = m.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        tokens = tokenized["input_ids"]
        offsets = tokenized["offset_mapping"]
        for segment, start in enumerate(range(0, max(1, len(tokens)), 512 - 48)):
            end = min(start + 512, len(tokens))
            piece = text[offsets[start][0]:offsets[end - 1][1]] if tokens else text
            vector = m.encode(f"title: {literal_text(title)} | text: {literal_text(piece)}", normalize_embeddings=True,
                              convert_to_numpy=True, show_progress_bar=False)
            if not np.isfinite(vector).all():
                raise RuntimeError("non-finite embedding")
            out.append({"input_index": index, "segment": segment, "text": piece, "embedding": vector.tolist()})
            if start + 512 >= len(tokens):
                break
    for item in out:
        if len(item["embedding"]) != DIMENSIONS or not all(math.isfinite(x) for x in item["embedding"]):
            raise RuntimeError("invalid embedding")
    return out


class Handler(BaseHTTPRequestHandler):
    def reply(self, status, data):
        raw = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path not in ("/health", "/v1/runtime/memory"):
            self.reply(404, {"error": "not found"}); return
        metrics = {}
        if model is not None:
            import torch
            metrics = {"cuda_allocated_gib": torch.cuda.memory_allocated() / 2**30,
                       "cuda_reserved_gib": torch.cuda.memory_reserved() / 2**30,
                       "cuda_peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}
        self.reply(200, {**metrics, "status": "ok", "busy": busy, "active": int(busy), "ready": model is not None,
                         "profile": PROFILE, "dimensions": DIMENSIONS, "device": "cuda", "dtype": "bf16",
                         "quiescing": quiescing, "queued": 0, "idle_for_seconds": max(0, time.monotonic() - last_used) if last_used else 0})

    def do_POST(self):
        global busy, last_used, quiescing
        if self.path == "/v1/runtime/reclaim-cache":
            if not lock.acquire(blocking=False):
                self.reply(409, {"error": "worker busy"}); return
            try:
                if model is not None and any(p.device.type != "cuda" for p in model.parameters()):
                    raise RuntimeError("checkpoint has CPU model parameters")
                checkpoint = Path(os.environ.get("MODEL_DIR", "/models")) / "model.safetensors"
                target = str(checkpoint.resolve())
                if any(target in line or "model.safetensors" in line for line in Path("/proc/self/maps").read_text().splitlines()):
                    raise RuntimeError("checkpoint is still mapped")
                for fd in Path("/proc/self/fd").iterdir():
                    try:
                        if str(fd.resolve()) == target:
                            raise RuntimeError("checkpoint is still open")
                    except FileNotFoundError:
                        pass
                with checkpoint.open("rb") as f:
                    os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                self.reply(200, {"status": "ok"})
            except Exception as e:
                self.reply(409, {"error": str(e)})
            finally:
                lock.release()
            return
        if self.path in ("/v1/runtime/quiesce", "/v1/runtime/resume"):
            if not lock.acquire(blocking=False):
                self.reply(409, {"error": "worker busy"}); return
            try:
                quiescing = self.path.endswith("quiesce")
                self.reply(200, {"status": "ok"})
            finally:
                lock.release()
            return
        if self.path != "/v1/encode":
            self.reply(404, {"error": "not found"}); return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 256 * 1024:
                raise ValueError("invalid request size")
            data = json.loads(self.rfile.read(size))
            inputs = data["inputs"]
            task = data["task"]
            if task not in ("query", "document") or not isinstance(inputs, list) or not 1 <= len(inputs) <= 2:
                raise ValueError("invalid task or batch size")
            if any(not isinstance(i, dict) or not isinstance(i.get("text"), str) or not i["text"].strip()
                   or len(i["text"]) > 32000 or len(i.get("title", "")) > 1000 for i in inputs):
                raise ValueError("invalid input")
            if task == "query" and len(inputs) != 1:
                raise ValueError("query batch must contain one input")
        except (ValueError, KeyError, TypeError) as e:
            self.reply(400, {"error": str(e)}); return
        if not lock.acquire(blocking=False):
            self.reply(409, {"error": "embedding worker busy"}); return
        if quiescing:
            lock.release()
            self.reply(409, {"error": "worker quiescing"}); return
        busy = True
        try:
            started = time.monotonic()
            result = encode(inputs, task)
            self.reply(200, {"profile": PROFILE, "dimensions": DIMENSIONS, "data": result,
                             "elapsed_seconds": time.monotonic() - started})
        except Exception as e:
            self.reply(503, {"error": str(e)})
        finally:
            last_used = time.monotonic()
            busy = False
            lock.release()


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", int(os.environ.get("PORT", "8701"))), Handler).serve_forever()
