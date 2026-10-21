"""Retain schema qualification, ordinary-generation regression and unified-memory evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics

from comparison_summary import memory


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    baseline=Path('/home/edp1096/.cache/model-download-jobs/velogb10-exllama-comparison-20261009/velo-full-vocab')
    baseline_cases={r['name']:r for r in json.loads((baseline/'suite.json').read_text())}
    single=a.root/'single';new_cases={r['name']:r for r in json.loads((single/'suite.json').read_text())}
    speed_names=[n for n in baseline_cases if n.startswith(('korean_','code_'))]
    clean_cases={r['name']:r for r in json.loads((a.root/'clean-speed/speed.json').read_text())}
    runs={}
    for label in ['single','batched']:
        out=a.root/label;raw=json.loads((out/'schema-results.json').read_text())
        valid=[r for r in raw if r['pass'] and r['response']['status']==200 and r['response'].get('headers',{}).get('x-json-schema-enforced')=='gpu' and r['response'].get('finish')=='stop']
        runs[label]={'HTTP':json.loads((out/'schema-summary.json').read_text()),'memory':memory(out/'memory.jsonl'),
            'command':json.loads((out/'command.json').read_text()),'exit':json.loads((out/'exit.json').read_text()),
            'normal_complete_schema_latency_seconds':{'median':statistics.median(r['response']['seconds'] for r in valid),
                'min':min(r['response']['seconds'] for r in valid),'max':max(r['response']['seconds'] for r in valid)},
            'memory_after_schema':json.loads((out/'memory-after-schema.json').read_text())}
    tests={}
    for name in ['unit-json-schema-compact.log','final-exl3_schema__tests.log','final-tokenizer__tests.log','final-server__.log','final-exl3_serve__.log','final-exl3_rope__.log','tokenizer-oracle-compact.log','gpu-mask.log','gpu-yarn.log']:
        text=(a.root/name).read_text();match=re.search(r'test result: ok\. (\d+) passed; (\d+) failed;',text)
        assert match,name;tests[name]={'passed':int(match[1]),'failed':int(match[2])}
    src=a.root/'source';binary=src/'target/release/gb10_inference'
    hashes={str(p.relative_to(src)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [binary,*sorted((src/'src/ptx').glob('*.ptx'))]}
    result={'base_commit':'a6ad23d60e082ff9cca2bb78adda4ecee771388c','patches':['velogb10-exl3-yarn.patch','velogb10-exl3-schema.patch'],
        'enabled':'By default on EXL3 single-GPU when response_format requests a supported JSON schema or json_object',
        'implementation':{'selection':'fp16 logits masked on GPU after penalties, before greedy or stochastic selection',
            'emission':'Every token checked before HTTP/SSE emission; stop only on completed JSON, length on exhausted budget',
            'ordinary_requests':'Existing CUDA graphs and MTP retained; no schema upload',
            'schema_requests':'Plain eager GPU decode, MTP bypassed, reasoning disabled, compact structural syntax',
            'tokenizer':'Same raw byte table as the streaming decoder; special-token pieces excluded',
            'unsupported':'Unsupported schema/topology, contradictory constraints, forced tool calls and stop strings return HTTP 400'},
        'scope':{'GPU':'one GB10','single_context':1048576,'batch_test_context':65536,'KV':'Q8','draft_vocabulary':'full','prefix_checkpoint_cap_GiB':4,
            'supported_schema_keywords':['type','properties','required','additionalProperties','items','minItems','maxItems','minLength','maxLength','minimum','maximum','exclusiveMinimum','exclusiveMaximum','enum (strings/integers)'],
            'limitations':['No real TP constraint support; requests explicitly rejected.','$ref, pattern, oneOf/anyOf/allOf and other unsupported keywords return 400.','Schema requests use non-reasoning plain GPU decode, not speculative decoding.','Bounded numeric generation chooses decimal spellings; successful output is schema-valid but not every equivalent spelling is generated.','Exhausted max_tokens can return incomplete JSON with finish_reason length.']},
        'runs':runs,'tests':tests,'GPU_sampled_rows':144,'YaRN_GPU_regression_cases':93,'byte_level_random_paths':100,'real_BPE_random_paths':32,
        'ordinary_regression':{'summary':json.loads((single/'suite-summary.json').read_text()),
            'same_visible_speed_outputs':{n:clean_cases[n]['text']==baseline_cases[n]['text'] for n in speed_names},
            'speed':{kind:{'baseline':statistics.median(baseline_cases[n]['decode_tps'] for n in speed_names if n.startswith(kind)),
                           'new':statistics.median(clean_cases[n]['decode_tps'] for n in speed_names if n.startswith(kind))} for kind in ['korean_','code_']},
            'clean_speed_run':'Fresh process, same initial warmups, same saved request bodies, no concurrent HTTP requests.',
            'primary_speed_run_limit':'Supplemental type probes overlapped the first suite speed cases; use clean-speed metrics for comparison.',
            'known_unfixed': 'The pre-existing literal XML tool-example requests still generate calls; unchanged by response_format enforcement.'},
        'long_context':json.loads((single/'recall-320k.json').read_text()),
        'extra_type_checks':json.loads((single/'extra-types.json').read_text()),
        'baseline_memory':memory(baseline/'memory.jsonl'),
        'reproduction':json.loads((a.root/'patch-reproduction.json').read_text()),'binary_and_PTX_sha256':hashes,
        'initial_failed_controls':['single-initial-registration (missing runtime kernel registration, fixed)',
            'single-initial-fsm (non-viable JSON prefixes, fixed)','single-initial-whitespace (whitespace budget exhaustion, fixed)'],
        'artifacts':str(a.root),'production':'Existing Talk/ExLlama configuration and original model weights unchanged'}
    assert all(x['HTTP']['passed']==x['HTTP']['total'] and not x['HTTP']['failed'] for x in runs.values())
    assert result['long_context']['pass']
    assert all(r['pass'] for r in result['extra_type_checks'])
    assert all(result['ordinary_regression']['same_visible_speed_outputs'].values())
    a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'HTTP':{k:v['HTTP'] for k,v in runs.items()},'memory':{k:v['memory'] for k,v in runs.items()},'speed':result['ordinary_regression']['speed']},ensure_ascii=False))


if __name__=='__main__':main()
