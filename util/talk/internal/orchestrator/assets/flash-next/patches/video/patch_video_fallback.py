"""Add a narrow, no-transcode fallback for Decord metadata failures."""
from pathlib import Path
import shutil

path = Path('/sgl-workspace/sglang/python/sglang/srt/utils/video_decoder.py')
shutil.copyfile(Path(__file__).with_name('video_fallback.py'), path.with_name('video_fallback.py'))
source = path.read_text()
if 'from .video_fallback import read_decord_or_pyav' not in source:
    old = '            from decord import VideoReader, cpu'
    assert source.count(old) == 1
    source = source.replace(old, old + '\n            from .video_fallback import read_decord_or_pyav')
    for expr in ['VideoReader(tmp_path, ctx=cpu(0))', 'VideoReader(source, ctx=cpu(0))']:
        assert source.count(expr) == 1
        source = source.replace(expr, expr.replace('VideoReader(', 'read_decord_or_pyav('))
    path.write_text(source)
print('Decord metadata fallback installed (PyAV; no transcoding)')
