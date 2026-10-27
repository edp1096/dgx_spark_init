"""Synchronize canonical speech helpers and voice assets into the Shorts context."""
from pathlib import Path
import shutil

source = Path(__file__).resolve().parent
target = source.parent / 'extra_shorts_service'
shutil.copy2(source / 'voice_catalog.py', target / 'voice_catalog.py')
shutil.copy2(source / 'speech.py', target / 'speech.py')
shutil.copytree(source / 'voices', target / 'voices', dirs_exist_ok=True)
for name in ['reference.wav', 'reference-uncle-fu.wav']:
    shutil.copy2(source / name, target / name)
