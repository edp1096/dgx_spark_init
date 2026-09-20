"""Measure allocated capacity separately from actual long-input retrieval."""
import argparse
import json
import pathlib
import random
import time
import urllib.error
import urllib.request

BASE = 'http://127.0.0.1:8016'
MODEL = 'qwen38-qad-sglang-trial'
FILLER = 'The archive entry contains routine information.\n'


def request(path, data=None, timeout=7200):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(BASE+path,
        data=json.dumps(data).encode() if data is not None else None,
        headers={'Content-Type': 'application/json'})
    with opener.open(req, timeout=timeout) as response:
        if path == '/flush_cache':
            return response.read().decode()
        return json.load(response)


def make_prompt(repetitions, diverse=False):
    # Different field values from the shorter operational probe; deterministic.
    if diverse:
        words = ('river stone cloud copper forest window garden bridge paper clock '
                 'silver amber blue green red white black yellow orange purple '
                 'house door table chair room floor wall roof road path '
                 'hill mountain valley lake sea beach island field grass tree '
                 'leaf branch root flower seed fruit apple pear peach grape '
                 'bird fish horse sheep lion tiger bear wolf fox deer '
                 'sun moon star light shadow morning evening night spring winter '
                 'summer autumn rain snow wind storm water fire earth sand '
                 'book letter page pencil bottle basket box bowl plate cup '
                 'shirt coat shoe hat bag belt rope wire wheel stone').split()
        rng = random.Random(20260919)
        segments = [''.join(' '.join(rng.choices(words,k=8))+'.\n'
                            for _ in range(repetitions)) for _ in range(3)]
    else:
        segments = [FILLER * repetitions]*3
    return ('Read this archive and remember its two named fields.\n' + segments[0] +
            '\nFIELD_ALPHA = MAPLE\n' + segments[1] + '\nFIELD_BETA = COMET\n' +
            segments[2] + '\nReturn the values of FIELD_ALPHA and FIELD_BETA, in that order, '
            'as two uppercase words separated by a comma. Nothing else.')


def tokenize(repetitions, diverse=False):
    return request('/v1/tokenize', dict(model=MODEL,
        messages=[dict(role='user', content=make_prompt(repetitions,diverse))],
        chat_template_kwargs={'enable_thinking': False}))['tokens']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=pathlib.Path, required=True)
    parser.add_argument('--targets', default='65536,262144,524288')
    parser.add_argument('--diverse', action='store_true')
    args = parser.parse_args()
    info = request('/server_info')
    args.output.with_name('server-info.json').write_text(json.dumps(info, indent=2))
    capacity = info['max_total_num_tokens']
    input_limit = info['max_req_input_len']
    results = [dict(name='capacity', configured_context=info['context_length'],
                    allocated_tokens=capacity, max_req_input_len=input_limit)]
    def save():
        args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    save()
    print(f'Configured context={info["context_length"]}, allocated KV={capacity}, request input limit={input_limit}', flush=True)
    if info['context_length'] > capacity:
        print('Configured context exceeds allocated KV capacity; it is NOT a validated usable context length.', flush=True)
    # Leave output and speculative pages available. A budget below the next
    # target is recorded as such, never rounded up to a supported context.
    maximum = min(input_limit-64, capacity-64)
    targets = sorted(set(int(t)-128 for t in args.targets.split(',')
                         if 8192<int(t)-128<=maximum))
    assert targets, ('No requested test length fits the measured capacity', capacity)
    failures = []
    for target in targets:
        request('/flush_cache', timeout=30)
        repeats = max(1,(target-64)//24)
        ids = tokenize(repeats,args.diverse)
        for _ in range(6):
            if len(ids)<=target and target-len(ids)<48:
                break
            per_repeat = (len(ids)-64)/repeats
            repeats = max(1,repeats+int((target-len(ids))/per_repeat)-int(len(ids)>target))
            ids = tokenize(repeats,args.diverse)
        assert len(ids)<=target and target-len(ids)<48,(target,len(ids))
        print(f'Beginning actual prompt_tokens={len(ids)} retrieval', flush=True)
        start = time.monotonic()
        response = request('/generate', dict(input_ids=ids,
            sampling_params=dict(temperature=0,max_new_tokens=64),stream=False))
        elapsed = time.monotonic()-start
        answer = response.get('text','').strip().replace(' ','')
        meta = response.get('meta_info',{})
        result = dict(name='context_retrieval',target_tokens=target,
            input_tokens=len(ids),seconds=elapsed,response=response,diverse=args.diverse,
            correct=answer=='MAPLE,COMET')
        results.append(result);save()
        print(f'Finished {len(ids)} tokens in {elapsed:.1f}s: answer={answer!r}, correct={result["correct"]}',flush=True)
        assert meta.get('prompt_tokens')==len(ids),meta
        if not result['correct']:
            failures.append(len(ids))
    if failures:
        raise AssertionError(f'Long-input retrieval failed at {failures}; allocation alone is not qualification')


if __name__ == '__main__':
    main()
