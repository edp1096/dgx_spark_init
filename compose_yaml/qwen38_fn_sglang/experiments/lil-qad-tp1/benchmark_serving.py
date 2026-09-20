"""Sequential OpenAI SSE workload shared by the isolated SGLang/vLLM trials.

Exact same prompt bytes, temperature, output budget and thinking policy.
Engine memory-pool and MTP settings are recorded separately, not equated.
"""
import argparse
import json
from pathlib import Path
import time
import urllib.request


def run(output, engine='sglang'):
    base, model = {'sglang': ('http://127.0.0.1:8016','qwen38-qad-sglang-trial'),
                   'vllm': ('http://127.0.0.1:8017','qwen38-qad-vllm-trial')}[engine]
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def reset():
        path = '/flush_cache' if engine == 'sglang' else '/reset_prefix_cache'
        req = urllib.request.Request(base+path, method='GET' if engine == 'sglang' else 'POST')
        with opener.open(req, timeout=60) as response:
            assert response.status == 200
            response.read()
    results = []
    def sample(repetitions, name):
        prompt = ('The following background is irrelevant to the counting task.\n'
                  + 'The archive entry contains routine information.\n' * repetitions
                  + '\nWrite integers 1 through 1000 in ascending order, comma separated. '
                  'Keep writing until 1000. No explanation, code or ellipses.')
        body = dict(model=model, messages=[dict(role='user', content=prompt)],
            temperature=0, max_tokens=256, stream=True,
            stream_options=dict(include_usage=True),
            chat_template_kwargs=dict(enable_thinking=False))
        req = urllib.request.Request(base+'/v1/chat/completions',
            data=json.dumps(body).encode(), headers={'Content-Type':'application/json'})
        start = time.monotonic()
        first = None; text = ''; usage = None; done = False; finish = None
        with opener.open(req, timeout=1800) as response:
            for line in response:
                if not line.startswith(b'data: '): continue
                payload = line[6:].strip()
                if payload == b'[DONE]': done=True; break
                event = json.loads(payload)
                if event.get('usage'): usage=event['usage']
                for choice in event.get('choices',[]):
                    piece=choice.get('delta',{}).get('content') or ''
                    if piece and first is None: first=time.monotonic()-start
                    text += piece
                    if choice.get('finish_reason'): finish=choice['finish_reason']
        elapsed = time.monotonic()-start
        result = dict(name=name, engine=engine, seconds=elapsed,
            first_content_seconds=first, usage=usage, finish_reason=finish,
            done=done, text=text)
        results.append(result)
        output.write_text(json.dumps(results, ensure_ascii=False, indent=2))
        assert done and usage and usage['completion_tokens']==256, result
        assert first is not None and text.strip().startswith('1'), result
        print(f'{engine} {name}: input={usage["prompt_tokens"]}, output=256, '
              f'TTFT={first:.3f}s, total={elapsed:.3f}s',flush=True)
    reset()
    sample(8, 'warmup')
    for repetitions in (128, 2048, 8000):
        reset()
        sample(repetitions, f'cold_{repetitions}')
        sample(repetitions, f'cached_{repetitions}')
    return results


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--engine', choices=['sglang','vllm'], default='sglang')
    args=parser.parse_args()
    run(args.output,args.engine)
