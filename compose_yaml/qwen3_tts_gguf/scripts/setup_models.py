"""Download the two pinned deployment GGUFs; verify before atomic replacement."""
import hashlib
import os
from pathlib import Path
import tempfile
import urllib.request

REVISION = 'b7ee2e8c7459c3bea99da23e3d178125a7d1713c'
FILES = {
    'qwen-talker-0.6b-customvoice-Q8_0.gguf': '4eb38675c736ed6ac72012846ac8d6ef80e5af8bc05726870f0b3a6569588519',
    'qwen-tokenizer-12hz-Q8_0.gguf': '1883beeed99348fc35e23dd225e9082f93f6f8c109330a33d935baa8acdbfd94',
}

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()

def main():
    directory = Path(os.environ.get('QWEN_TTS_MODEL_DIR', str(Path.home() / '.cache/qwen3-tts'))).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        target = directory / name
        if target.is_file() and digest(target) == expected:
            print('Verified:', name)
            continue
        url = f'https://huggingface.co/Serveurperso/Qwen3-TTS-GGUF/resolve/{REVISION}/{name}'
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=directory, prefix=name + '.', suffix='.partial', delete=False) as output:
                temporary = Path(output.name)
                with urllib.request.urlopen(url, timeout=120) as response:
                    while block := response.read(1024 * 1024): output.write(block)
            if digest(temporary) != expected: raise RuntimeError('SHA256 mismatch: ' + name)
            temporary.chmod(0o644)
            temporary.replace(target)
            print('Downloaded and verified:', name)
        finally:
            if temporary is not None: temporary.unlink(missing_ok=True)

if __name__ == '__main__': main()
