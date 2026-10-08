#!/usr/bin/env python3
"""Build a TP1 draft shortlist from synthetic calibration output.

Keep all complete Hangul tokens and special tokens. Rank observed calibration
tokens by frequency, then fill by tokenizer ID. This changes drafting only.
Run inside the serving image (tokenizers and torch are already installed).
"""
import argparse
import collections
import hashlib
import json
import pathlib

from tokenizers import Tokenizer


def byte_decoder():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    extra = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + extra)
            extra += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--size", type=int, default=65536)
    ap.add_argument("--verify", help="held-out result JSON; reports coverage only")
    args = ap.parse_args()
    raw = pathlib.Path(args.tokenizer).read_bytes()
    data = json.loads(raw)
    tokenizer = Tokenizer.from_file(args.tokenizer)
    inverse = byte_decoder()
    protected = set(range(256)) | {v["id"] for v in data["added_tokens"]}
    hangul = set()
    for text, ident in data["model"]["vocab"].items():
        try:
            decoded = bytes(inverse[c] for c in text).decode("utf-8")
        except (KeyError, UnicodeDecodeError):
            continue
        if any("\uac00" <= c <= "\ud7a3" for c in decoded):
            hangul.add(ident)
    protected.update(hangul)
    freq = collections.Counter()
    corpus = json.loads(pathlib.Path(args.corpus).read_text())
    for row in corpus["rows"]:
        for field in ["prompt", "content"]:
            freq.update(tokenizer.encode(row[field], add_special_tokens=False).ids)
    chosen = set(protected)
    valid = set(tokenizer.get_vocab().values())
    if len(chosen) > args.size or args.size > len(valid):
        raise ValueError("size must fit protected tokens and tokenizer vocabulary")
    for ident in sorted(freq, key=lambda ident: (-freq[ident], ident)) + sorted(valid):
        if len(chosen) >= args.size:
            break
        chosen.add(ident)
    ids = sorted(chosen)
    ranges = []
    for ident in ids:
        if ranges and ident == ranges[-1][1]:
            ranges[-1][1] += 1
        else:
            ranges.append([ident, ident+1])
    def coverage(counts):
        return sum(n for ident, n in counts.items() if ident in chosen) / max(1, sum(counts.values()))
    result = dict(version=1, size=len(ids), tokenizer_sha256=hashlib.sha256(raw).hexdigest(),
                  corpus_sha256=hashlib.sha256(pathlib.Path(args.corpus).read_bytes()).hexdigest(),
                  policy="special + byte tokens + all complete Hangul tokens + corpus frequency + ascending ID fill",
                  protected_hangul=len(hangul), calibration_coverage=coverage(freq), ranges=ranges)
    if args.verify:
        by_category = {}
        for row in json.loads(pathlib.Path(args.verify).read_text())["rows"]:
            by_category.setdefault(row["name"], collections.Counter()).update(tokenizer.encode(row["content"], add_special_tokens=False).ids)
        result["held_out_coverage"] = {k: coverage(v) for k,v in by_category.items()}
    out = pathlib.Path(args.output)
    out.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
    import torch
    torch.save(ids, str(out.with_suffix(".pt")))
    assert len(ids) == args.size and len(set(ids)) == args.size and set(ids) <= valid
    print(json.dumps({k:v for k,v in result.items() if k != "ranges"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
