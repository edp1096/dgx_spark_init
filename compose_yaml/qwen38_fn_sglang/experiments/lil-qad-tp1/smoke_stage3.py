"""Operational probes after the short vision/MTP checks: batching, SSE, QSA."""
import argparse
import concurrent.futures
import json
import pathlib
import re
import subprocess
import sys
import threading
import time
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
BASE = 'http://127.0.0.1:8016'


def call(prompt, stream=False):
    body = dict(model='qwen38-qad-sglang-trial',
                messages=[dict(role='user', content=prompt)],
                temperature=0, max_tokens=512, stream=stream,
                reasoning_effort='none')
    request = urllib.request.Request(BASE+'/v1/chat/completions',
        data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    start = time.monotonic()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=1800) as response:
        if not stream:
            result = json.load(response)
        else:
            chunks = []
            done = False
            for line in response:
                if not line.startswith(b'data: '):
                    continue
                payload = line[6:].strip()
                if payload == b'[DONE]':
                    done = True
                    break
                chunks.append(json.loads(payload))
            result = {'chunks': chunks, 'done': done}
    return {'seconds': time.monotonic()-start, 'response': result}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=pathlib.Path, required=True)
    args = parser.parse_args()
    results = json.loads(args.output.read_text()) if args.output.exists() else []

    def record(name, result):
        result['name'] = name
        results.append(result)
        args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2))

    barrier = threading.Barrier(2)
    def count(start):
        barrier.wait()
        return call(f'Write integers {start} through {start+59}, inclusive, ascending, comma separated. Nothing else.')
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [(start, pool.submit(count, start)) for start in (1,101)]
        for start, job in jobs:
            result = job.result()
            record(f'concurrent_count_{start}', result)
            choice = result['response']['choices'][0]
            assert choice['finish_reason'] == 'stop', choice
            assert re.findall(r'\d+', choice['message'].get('content') or '') == [str(x) for x in range(start,start+60)], choice

    result = call('7 더하기 9의 결과만 숫자로 답하세요.', stream=True)
    record('streaming', result)
    response = result['response']
    text = ''.join(c['choices'][0]['delta'].get('content') or '' for c in response['chunks'] if c.get('choices'))
    assert response['done'] and text.strip() == '16', response

    filler = 'The archive entry contains routine information.\n' * 600
    prompt = ('Read the archive and remember its two named fields.\n' + filler +
              '\nFIELD_ALPHA = CEDAR\n' + filler + '\nFIELD_BETA = ORBIT\n' +
              filler + '\nReturn the values of FIELD_ALPHA and FIELD_BETA, in that order, '
              'as two uppercase words separated by a comma. Nothing else.')
    result = call(prompt)
    record('long_qsa_retrieval', result)
    response = result['response']
    assert response['usage']['prompt_tokens'] > 8192, response['usage']
    message = response['choices'][0]['message']
    assert (message.get('content') or '').strip().replace(' ','') == 'CEDAR,ORBIT', message
    print('Concurrent requests, SSE and >8K QSA retrieval passed', flush=True)
    from smoke_prefix import run
    run(args.output)


if __name__ == '__main__':
    main()
