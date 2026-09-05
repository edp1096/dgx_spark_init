#!/usr/bin/env python3
"""Compile the actual cache update from a NeMo checkout and test its token alignment.

Usage: python3 scripts/test_longform_cache.py /path/to/NeMo-Speech.cpp
The unpatched pinned revision must fail; the patched revision must pass.
"""
import pathlib
import subprocess
import sys
import tempfile

source = (pathlib.Path(sys.argv[1]) / 'src/tts/magpietts/magpietts.cpp').read_text()
start = source.index('static bool\nsplice_longform_history_context(')
end = source.index('\nstatic ', start + 1)
splice = source[start:end]
end = source.index('\n            metrics.encoder_ms', source.index('history_text_context_len = 0;'))
start = source.rfind('                history_text_context = text_cond;', 0, end)
if start < 0:
    start = source.index('                // Token history spans all previous chunks')
update = source[start:end].rsplit('\n            }', 1)[0]
harness = r'''
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <vector>
SPLICE
int main() {
    for (int capacity : {64, 32}) {
        struct { int n_ctx; int n_embd; } h{capacity, 3};
        std::vector<float> history_text_context;
        int history_text_context_len = 0;
        int total = 0;
        // Long -> tiny -> tiny -> long reproduces a 20-needed/18-cached failure.
        for (int count : {24, 6, 9, 24, 32, 1, 2, 24, 32, 32}) {
            const std::vector<int> current_tokens(count, 0);
            const int history_len = std::min({total, count, 20, capacity-count});
            const int text_len = history_len + count;
            std::vector<float> text_cond(text_len*h.n_embd, -1.f);
            for (int i=0; i<count; ++i)
                for (int j=0; j<h.n_embd; ++j)
                    text_cond[(history_len+i)*h.n_embd+j] = (total+i)*10+j;
            if (!splice_longform_history_context(text_cond, text_len, count, h.n_embd,
                                                  history_text_context, history_text_context_len)) return 1;
            for (int i=0; i<text_len; ++i)
                for (int j=0; j<h.n_embd; ++j)
                    assert(text_cond[i*h.n_embd+j] == (total-history_len+i)*10+j);
UPDATE
            total += count;
            assert(history_text_context_len <= capacity);
            assert(history_text_context.size() == size_t(history_text_context_len*h.n_embd));
            for (int i=0; i<history_text_context_len; ++i)
                for (int j=0; j<h.n_embd; ++j)
                    assert(history_text_context[i*h.n_embd+j] == (total-history_text_context_len+i)*10+j);
        }
    }
    puts("longform cache alignment and bounded retention passed");
}
'''.replace('SPLICE', splice).replace('UPDATE', update)
with tempfile.TemporaryDirectory(prefix='magpie-cache-test-') as tmp:
    cpp = pathlib.Path(tmp) / 'test.cpp'
    exe = pathlib.Path(tmp) / 'test'
    cpp.write_text(harness)
    subprocess.run(['c++', '-std=c++17', '-fsanitize=address,undefined', '-g', str(cpp), '-o', str(exe)], check=True)
    subprocess.run([str(exe)], check=True)
