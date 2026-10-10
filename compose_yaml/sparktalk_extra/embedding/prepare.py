import os
from huggingface_hub import snapshot_download
MODEL = "google/embeddinggemma-2"
REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
if __name__ == "__main__":
    snapshot_download(MODEL, revision=REVISION, local_dir=os.environ.get("MODEL_DIR", "/models"),
                      ignore_patterns=["*.md", ".gitattributes"])
