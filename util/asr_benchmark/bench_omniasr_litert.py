#!/usr/bin/env python3
import argparse
import json
import math
import statistics
import time
import unicodedata
from pathlib import Path

import numpy as np
import sentencepiece as spm
import soundfile as sf
from ai_edge_litert.interpreter import Interpreter


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return "".join(character for character in text if character.isalnum())


def edit_distance(reference: str, hypothesis: str) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, reference_character in enumerate(reference, 1):
        current = [row]
        for column, hypothesis_character in enumerate(hypothesis, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1]
                    + (reference_character != hypothesis_character),
                )
            )
        previous = current
    return previous[-1]


def read_pcm16_mono(path: Path) -> np.ndarray:
    samples, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if sample_rate != 16_000 or samples.ndim != 1:
        raise ValueError(f"expected mono 16 kHz audio: {path}")
    return samples


class OmniASRLiteRT:
    def __init__(self, model: Path, tokenizer: Path, threads: int):
        self.interpreter = Interpreter(model_path=str(model), num_threads=threads)
        self.interpreter.allocate_tensors()
        self.input = self.interpreter.get_input_details()[0]
        self.output = self.interpreter.get_output_details()[0]
        self.tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer))
        self.input_samples = int(self.input["shape"][-1])
        self.output_frames = int(self.output["shape"][-2])
        self.blank_id = self.tokenizer.pad_id()

    def _decode_chunk(self, samples: np.ndarray) -> str:
        valid_samples = len(samples)
        if valid_samples < self.input_samples:
            samples = np.pad(samples, (0, self.input_samples - valid_samples))
        mean = float(samples[:valid_samples].mean()) if valid_samples else 0.0
        std = float(samples[:valid_samples].std()) if valid_samples else 1.0
        samples = (samples - mean) / max(std, 1e-7)
        self.interpreter.set_tensor(self.input["index"], samples.reshape(1, -1))
        self.interpreter.invoke()
        logits = self.interpreter.get_tensor(self.output["index"])[0]
        valid_frames = min(
            self.output_frames,
            max(1, math.ceil(valid_samples * self.output_frames / self.input_samples)),
        )
        frame_ids = np.argmax(logits[:valid_frames], axis=-1).tolist()
        token_ids = []
        previous = None
        for token_id in frame_ids:
            if token_id != previous and token_id != self.blank_id:
                token_ids.append(int(token_id))
            previous = token_id
        return self.tokenizer.decode(token_ids).strip()

    def transcribe(self, samples: np.ndarray) -> str:
        chunks = [
            samples[offset : offset + self.input_samples]
            for offset in range(0, len(samples), self.input_samples)
        ]
        return " ".join(self._decode_chunk(chunk) for chunk in chunks).strip()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, round(fraction * (len(ordered) - 1)))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    records = [
        json.loads(line)
        for line in (args.corpus / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    start = time.perf_counter()
    model = OmniASRLiteRT(args.model, args.tokenizer, args.threads)
    load_seconds = time.perf_counter() - start

    # Warm up the fixed-shape graph without counting it.
    model.transcribe(read_pcm16_mono(args.corpus / records[0]["file"]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    total_edits = 0
    total_characters = 0
    for record in records:
        samples = read_pcm16_mono(args.corpus / record["file"])
        start = time.perf_counter()
        text = model.transcribe(samples)
        latency = time.perf_counter() - start
        reference = normalize(record["reference"])
        hypothesis = normalize(text)
        edits = edit_distance(reference, hypothesis)
        total_edits += edits
        total_characters += len(reference)
        result = {
            **record,
            "text": text,
            "latency_seconds": latency,
            "edits": edits,
            "reference_characters": len(reference),
        }
        results.append(result)
        print(
            f"{record['file']} latency={latency:.3f}s "
            f"cer={edits / max(1, len(reference)):.4f} text={text}"
        )

    with args.output.open("w", encoding="utf-8") as output:
        for result in results:
            output.write(json.dumps(result, ensure_ascii=False) + "\n")

    latencies = [result["latency_seconds"] for result in results]
    audio_seconds = sum(result["duration_seconds"] for result in results)
    inference_seconds = sum(latencies)
    print(f"samples={len(results)}")
    print(f"load_seconds={load_seconds:.6f}")
    print(f"cer={total_edits / max(1, total_characters):.6f}")
    print(f"latency_mean_seconds={statistics.mean(latencies):.6f}")
    print(f"latency_median_seconds={statistics.median(latencies):.6f}")
    print(f"latency_p95_seconds={percentile(latencies, 0.95):.6f}")
    print(f"audio_seconds={audio_seconds:.3f}")
    print(f"inference_seconds={inference_seconds:.3f}")
    print(f"rtf={inference_seconds / audio_seconds:.6f}")
    print(f"realtime_factor_x={audio_seconds / inference_seconds:.3f}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
