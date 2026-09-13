"""Check long-prefix continuation and invalidation across prefill windows."""
import argparse
import json
from pathlib import Path
import time
import requests

p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
base = 'http://127.0.0.1:8010'
secret = 'quartz-lantern-438'
text = '\n'.join(
    f'Record {i:04d}: '+(f'the secret phrase is {secret}.' if i == 777 else
                        f'the archived label is blue and its numeric value is {i % 29}.')
    for i in range(900))
initial = {'role': 'user', 'content': text+'\nRead the records. Return only window-ready.'}
salt = f'window-continuation-{time.time_ns()}'
results = []


def call(messages, expected, reuse=False):
    reply = requests.post(base+'/v1/chat/completions', timeout=600, json={
        'model': 'deepseek-v4.1-flash', 'messages': messages,
        'temperature': 0, 'seed': 42, 'max_completion_tokens': 32,
        'chat_template_kwargs': {'thinking': False}, 'cache_salt': salt,
    })
    reply.raise_for_status()
    data = reply.json()
    choice = data['choices'][0]
    actual = choice['message']['content']
    assert actual.strip().strip('`". *') == expected and choice['finish_reason'] == 'stop', choice
    cached = data['usage']['prompt_tokens_details']['cached_tokens']
    if reuse:
        assert cached > 8192, data['usage']
    results.append({'expected': expected, 'actual': actual, 'usage': data['usage'],
                    'metrics': data.get('metrics')})
    a.output.write_text(json.dumps(results, indent=2)+'\n')
    print(expected, data['usage'], flush=True)
    return actual


answer = call([initial], 'window-ready')
assert results[0]['usage']['prompt_tokens'] > 8192
call([initial, {'role': 'assistant', 'content': answer},
      {'role': 'user', 'content': 'What is the exact secret phrase in record 0777? Return only that phrase.'}],
     secret, reuse=True)
changed = text.replace(secret, 'amber-orbit-962')
call([{'role': 'user', 'content': changed+'\nReturn only the secret phrase in record 0777.'}],
     'amber-orbit-962', reuse=True)
call([{'role': 'user', 'content': 'Return only short-ok.'}], 'short-ok')
print('PREFILL_WINDOW_CONTINUATION_PASS', flush=True)
