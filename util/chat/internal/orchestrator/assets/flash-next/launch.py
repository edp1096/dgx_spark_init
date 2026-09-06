"""Optional, tokenizer-checked draft shortlist for the existing SGLang server."""
import hashlib
import json
import os
from pathlib import Path
import sys


def server_arguments(arguments, mode, assets=Path(__file__).parent):
    if mode not in ("off", "ko64k"):
        raise ValueError("SPARKTALK_FLASH_NEXT_DRAFT_VOCAB must be off or ko64k")
    arguments = list(arguments)
    if mode == "off" or "--help" in arguments or any(a == "--speculative-token-map" or a.startswith("--speculative-token-map=") for a in arguments):
        return arguments
    def option(name):
        for i, arg in enumerate(arguments):
            if arg == name and i+1 < len(arguments):
                return arguments[i+1]
            if arg.startswith(name + "="):
                return arg.split("=", 1)[1]
        return None
    model = option("--tokenizer-path") or option("--model-path")
    if not model:
        raise ValueError("ko64k requires a local model/tokenizer path")
    metadata = json.loads((assets / "draft_vocab.json").read_text())
    tokenizer = Path(model) / "tokenizer.json"
    if hashlib.sha256(tokenizer.read_bytes()).hexdigest() != metadata["tokenizer_sha256"]:
        raise ValueError("ko64k tokenizer mismatch: use off or rebuild the shortlist for this tokenizer")
    shortlist = assets / "draft_vocab.pt"
    if not shortlist.is_file():
        raise ValueError("ko64k draft shortlist is missing from the runtime image")
    return arguments + ["--speculative-token-map", str(shortlist)]


if __name__ == "__main__":
    args = server_arguments(sys.argv[1:], os.environ.get("SPARKTALK_FLASH_NEXT_DRAFT_VOCAB", "off"))
    os.execv(sys.executable, [sys.executable, "-m", "sglang.launch_server", *args])
