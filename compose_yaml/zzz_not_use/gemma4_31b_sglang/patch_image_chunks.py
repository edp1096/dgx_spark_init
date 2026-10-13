"""Keep Gemma's bidirectional image spans within one prefill chunk."""
from pathlib import Path
import sys


HELPER = '''
def _gemma_image_chunk_length(req, length, page_size):
    mm = req.multimodal_inputs
    if mm is None:
        return length
    start = len(req.prefix_indices)
    end = start + length
    for item in mm.mm_items:
        if not item.is_image():
            continue
        for first, last in item.offsets or []:
            if first < start <= last:
                raise RuntimeError("Gemma image prefix starts inside an image span")
            if first < end <= last:
                # Cut before the image, preserving the allocator's alignment.
                end = first // page_size * page_size
                if end <= start:
                    raise RuntimeError("Gemma image exceeds the available prefill chunk")
    return end - start

'''


def patch(path):
    source = path.read_text()
    if 'def _gemma_image_chunk_length(' in source:
        return
    edits = [
        ('logger = logging.getLogger(__name__)\n', 'logger = logging.getLogger(__name__)\n' + HELPER),
        ('        new_len = min(cand_extend_input_len, _rem_tokens)\n'
         '        req.set_extend_range(len(req.prefix_indices), len(req.prefix_indices) + new_len)',
         '        new_len = min(cand_extend_input_len, _rem_tokens)\n'
         '        new_len = _gemma_image_chunk_length(req, new_len, self.page_size)\n'
         '        req.set_extend_range(len(req.prefix_indices), len(req.prefix_indices) + new_len)'),
        ('            trunc_len = self.rem_chunk_tokens\n',
         '            trunc_len = _gemma_image_chunk_length(req, self.rem_chunk_tokens, self.page_size)\n'),
        ('                now_input_len = trunc_len + len(req.prefix_indices)\n',
         '                trunc_len = _gemma_image_chunk_length(req, trunc_len, self.page_size)\n'
         '                now_input_len = trunc_len + len(req.prefix_indices)\n'),
    ]
    for before, after in edits:
        section_start, section_end = 0, len(source)
        if 'new_len = min' in before:
            section_start = source.index('    def add_chunked_req(')
            section_end = source.index('    @contextmanager', section_start)
        section = source[section_start:section_end]
        count = section.count(before)
        if count != 1:
            raise RuntimeError(f'Unexpected SGLang image-chunk patch site count: {count}')
        source = source[:section_start] + section.replace(before, after) + source[section_end:]
    compile(source, str(path), 'exec')
    path.write_text(source)


if __name__ == '__main__':
    patch(Path(sys.argv[1] if len(sys.argv) > 1 else
               '/sgl-workspace/sglang/python/sglang/srt/managers/schedule_policy.py'))
