"""Reproduce and verify the qualified Q5_K release without loading CUDA."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

MODEL = "nemotron-3.5-asr-streaming-0.6b.q5_k.gguf"
MODEL_SHA256 = "f0dab30ca22a606c2ab9a65efae88421851c6ca4ab01160708ab733b4467ce58"
MODEL_BYTES = 528061920
Q8 = "nemotron-3.5-asr-streaming-0.6b.q8_0.gguf"
Q8_SHA256 = "3fc991d3badad7277c11030a7519832cddaf2057aafed6d4b25147e953a070b1"
F16 = "nemotron-3.5-asr-streaming-0.6b.f16.gguf"
F16_SHA256 = "cc5fc1b6e05fdb905c4d9bc01842e1d297650cc39e117334bf16d64eb2b55fc0"
NEMO = "nemotron-3.5-asr-streaming-0.6b.nemo"
NEMO_SHA256 = "210214ed94039bf6bfbb9a047c7fa289628db75b103e2bf6381fa78285436a74"


def digest(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def matches(path, expected, size=None):
    try:
        return (size is None or path.stat().st_size == size) and digest(path) == expected
    except OSError:
        return False


def generate(source, template, output):
    import numpy as np
    import gguf
    sys.path.insert(0, "/src")
    from conversion.quantization import quantize

    original = gguf.GGUFReader(template)
    floats = gguf.GGUFReader(source)
    tensors = {tensor.name: tensor for tensor in floats.tensors}
    if {tensor.name for tensor in original.tensors} != set(tensors):
        raise RuntimeError("F16 and qualified Q8 tensor names do not match")
    writer = gguf.GGUFWriter(output, original.fields["general.architecture"].contents(), use_temp_file=True)
    try:
        for key, field in original.fields.items():
            if key.startswith("GGUF.") or key == "general.architecture":
                continue
            value = field.contents()
            if key == "general.file_type":
                value = 17  # GGUF MOSTLY_Q5_K
            elif key == "general.name":
                value = "nemotron-3.5-asr-streaming-0.6b.q5_k"
            subtype = field.types[-1] if field.types[0] == gguf.GGUFValueType.ARRAY else None
            writer.add_key_value(key, value, field.types[0], subtype)
        for tensor in original.tensors:
            float_tensor = tensors[tensor.name]
            if not np.array_equal(tensor.shape, float_tensor.shape):
                raise RuntimeError("F16 tensor shape mismatch: " + tensor.name)
            if tensor.tensor_type == gguf.GGMLQuantizationType.Q8_0:
                values = np.asarray(float_tensor.data, dtype=np.float32)
                if values.shape[-1] % 256:
                    # Six predictor/joiner matrices cannot use K-quant blocks.
                    packed, kind = values.astype(np.float16), gguf.GGMLQuantizationType.F16
                else:
                    packed, kind = quantize(values, gguf.GGMLQuantizationType.Q5_K), gguf.GGMLQuantizationType.Q5_K
                writer.add_tensor(tensor.name, packed, raw_dtype=kind)
            else:
                # Preserve tokenizer, position tables, norms, biases and convolutions.
                writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
    finally:
        writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("/models"))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    target = args.model_dir / MODEL
    if matches(target, MODEL_SHA256, MODEL_BYTES):
        print("Qualified Q5_K ready: " + str(target), flush=True)
        return
    if args.check:
        raise SystemExit(1)
    if not matches(args.model_dir / Q8, Q8_SHA256):
        raise RuntimeError("The pinned source Q8 checkpoint is missing or corrupt")
    with tempfile.TemporaryDirectory(prefix="nemotron-q5-") as temp:
        source = args.model_dir / F16
        if not matches(source, F16_SHA256):
            nemo = args.model_dir / NEMO
            if not matches(nemo, NEMO_SHA256):
                raise RuntimeError("The pinned NeMo source checkpoint is missing or corrupt")
            source = Path(temp) / F16
            # The converter runs as the host user; installed Python packages
            # are read-only. Librosa/Numba must cache in this private temp dir.
            environment = dict(os.environ, NUMBA_CACHE_DIR=str(Path(temp) / "numba-cache"))
            subprocess.run([sys.executable, "/src/convert_model.py", str(nemo),
                            "--outtype", "f16", "--outfile", str(source)], check=True, env=environment)
        # Keep the existing target intact until a complete, verified replacement exists.
        with tempfile.NamedTemporaryFile(prefix=MODEL + ".", suffix=".partial", dir=args.model_dir,
                                         delete=False) as file:
            partial = Path(file.name)
        try:
            generate(source, args.model_dir / Q8, partial)
            if not matches(partial, MODEL_SHA256, MODEL_BYTES):
                raise RuntimeError("Converted Q5_K does not match the qualified release SHA-256")
            with partial.open("rb") as file:
                os.fsync(file.fileno())
            os.replace(partial, target)
        finally:
            partial.unlink(missing_ok=True)
    print(json.dumps({"model": MODEL, "bytes": MODEL_BYTES, "sha256": MODEL_SHA256}), flush=True)


if __name__ == "__main__":
    main()
