"""Retain exact rendered IDs, compare MTP, then run an independent EXL3 engine."""
import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path


def post(url, route, body):
    req = urllib.request.Request(url.rstrip("/") + route, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def native(args):
    from bench import cases
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    chosen = {"prose-ko", "code-en", "arithmetic-net-price", "probability"}
    for mode in ["off", "medium"]:
        for case in cases():
            if case["id"] not in chosen:
                continue
            f = out / f'{case["id"]}-{mode}.json'
            if f.exists():
                continue
            body = {"model": args.model, "messages": [{"role": "user", "content": case["prompt"]}], "temperature": 0, "seed": 99, "max_tokens": 96, "ignore_eos": True, "reasoning_effort": "none" if mode == "off" else mode, "chat_template_kwargs": {"enable_thinking": mode != "off"}}
            ids = post(args.url, "/tokenize", body)["tokens"]
            plain = post(args.url, "/v1/chat/completions", dict(body, draft=False))
            drafted = post(args.url, "/v1/chat/completions", dict(body, draft=True))
            same = plain["tensorfold"]["token_sha256"] == drafted["tensorfold"]["token_sha256"]
            row = {"case": case["id"], "mode": mode, "request": body, "input_ids": ids, "plain": plain, "drafted": drafted, "drafted_equals_plain": same}
            f.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n")
            print(json.dumps({"case": case["id"], "mode": mode, "mtp_equal": same, "plain_tg": plain["tensorfold"]["tokens_per_second"], "mtp_tg": drafted["tensorfold"]["tokens_per_second"]}), flush=True)
            if not same:
                raise RuntimeError("MTP/plain token mismatch; retained both responses")


def reference(args):
    import torch
    from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job, GreedySampler
    config = Config.from_directory(args.checkpoint)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=16384)
    start = time.monotonic()
    model.load(progressbar=True)
    tokenizer = Tokenizer.from_config(config)
    generator = Generator(model=model, cache=cache, tokenizer=tokenizer, max_batch_size=1, max_chunk_size=2048, recurrent_cache_size=512 * 1024**2)
    print("REFERENCE_LOADED", round(time.monotonic() - start, 2), flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for f in sorted(Path(args.inputs).glob("*.json")):
            row = json.loads(f.read_text())
            if "input_ids" not in row:
                continue
            dest = out / f.name
            if dest.exists():
                continue
            job = Job(input_ids=torch.tensor([row["input_ids"]], dtype=torch.long), max_new_tokens=96, sampler=GreedySampler(), stop_conditions=[], decode_special_tokens=True)
            generator.enqueue(job)
            tokens = []
            text = []
            start = time.monotonic()
            while generator.num_remaining_jobs():
                for result in generator.iterate():
                    if "token_ids" in result:
                        tokens.extend(result["token_ids"].reshape(-1).tolist())
                    text.append(result.get("text", ""))
            sha = hashlib.sha256(",".join(map(str, tokens)).encode()).hexdigest()
            native_sha = row["plain"]["tensorfold"]["token_sha256"]
            result = {"case": row["case"], "mode": row["mode"], "input_ids": row["input_ids"], "output_ids": tokens, "text": "".join(text), "token_sha256": sha, "native_sha256": native_sha, "equal": sha == native_sha, "seconds": time.monotonic() - start}
            dest.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
            print(json.dumps({"case": row["case"], "mode": row["mode"], "tokens": len(tokens), "equal": sha == native_sha, "text": result["text"][:140]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="action", required=True)
    n = sub.add_parser("native")
    n.add_argument("--url", required=True)
    n.add_argument("--model", required=True)
    n.add_argument("--out", required=True)
    r = sub.add_parser("reference")
    r.add_argument("--checkpoint", required=True)
    r.add_argument("--inputs", required=True)
    r.add_argument("--out", required=True)
    args = p.parse_args()
    (native if args.action == "native" else reference)(args)
