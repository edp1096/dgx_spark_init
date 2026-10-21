"""Add the thinking switch missing from the NVIDIA GLM template, in runtime cache."""
import sys
from pathlib import Path


ORIGINAL = "<|assistant|>{{- '<think>' -}}"
SWITCHED = "<|assistant|>{{- '<think>' if enable_thinking is not defined or enable_thinking else '<think></think>' -}}"


def prepare(source: str) -> str:
    if source.count(SWITCHED) == 1 and ORIGINAL not in source:
        return source
    if source.count(ORIGINAL) != 1:
        raise ValueError("Unrecognized GLM generation prompt; review the upstream chat template")
    return source.replace(ORIGINAL, SWITCHED)


if __name__ == "__main__":
    model, destination = map(Path, sys.argv[1:])
    result = prepare((model / "chat_template.jinja").read_text())
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(result)
    temporary.replace(destination)
