"""Compare cold/cached prefix branches and a real two-turn conversation."""
import json
import time
from probe_context import request, MODEL


def run(output):
    results = json.loads(output.read_text()) if output.exists() else []
    def record(name, result, elapsed):
        results.append(dict(name=name, seconds=elapsed, response=result))
        output.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    prefix = ('Remember the exact two archive values. FIELD_ALPHA = MAPLE; FIELD_BETA = COMET.\n'
              + 'This ordinary archive entry has no other named fields.\n' * 173)
    def generate(field, name):
        prompt = prefix + f'\nReturn only the uppercase value of FIELD_{field}, no explanation.'
        tokens = request('/v1/tokenize', dict(model=MODEL,
            messages=[dict(role='user', content=prompt)],
            chat_template_kwargs={'enable_thinking': False}))['tokens']
        start = time.monotonic()
        result = request('/generate', dict(input_ids=tokens,
            sampling_params=dict(temperature=0, max_new_tokens=32)))
        record(name, result, time.monotonic()-start)
        assert result['text'].strip() == {'ALPHA':'MAPLE','BETA':'COMET'}[field], result
        return result
    request('/flush_cache')
    generate('BETA', 'prefix_cold_branch')
    request('/flush_cache')
    generate('ALPHA', 'prefix_seed')
    cached = generate('BETA', 'prefix_cached_branch')
    assert cached['meta_info']['cached_tokens'] > 512, cached
    messages = [dict(role='user', content=prefix + '\nReply only READY.')]
    for turn, expected in enumerate(('READY','MAPLE,COMET')):
        start = time.monotonic()
        result = request('/v1/chat/completions', dict(model=MODEL, messages=messages,
            temperature=0, max_tokens=64, reasoning_effort='none'))
        record(f'multiturn_{turn+1}', result, time.monotonic()-start)
        answer = result['choices'][0]['message']['content']
        assert answer.strip().replace(' ','') == expected, result
        messages.extend([dict(role='assistant', content=answer),
            dict(role='user', content='Return FIELD_ALPHA and FIELD_BETA as uppercase words separated by a comma. Nothing else.')])
    print('Cold/cached prefix branches and two-turn conversation passed', flush=True)
