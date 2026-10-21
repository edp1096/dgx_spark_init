from pathlib import Path
import json,statistics,hashlib,shutil,difflib,subprocess,sys
r=Path(__file__).parent
assert (r/'ALL_COMPARISONS_FINISHED').exists()
depths=[9,18,21,24]
cs=json.loads((r/'cases.json').read_text());expected={x['name']:hashlib.sha256(json.dumps(x['input_ids']).encode()).hexdigest() for x in cs}
summary={}
for d in depths:
 arm=f'splitkv-b4-high-{d}';a=r/arm
 assert (a/'COMPLETE').exists()
 rows=json.loads((a/'results.json').read_text());assert len(rows)==8
 for x in rows:
  assert x['input_sha256']==expected[x['name']]
  assert x['replacement_characters']==0
  m=x['meta_info'];h=m['spec_correct_drafts_histogram'];v=m['spec_verify_ct']
  assert sum(h)==v
  assert sum(i*c for i,c in enumerate(h))==m['spec_num_correct_drafts']
  assert m['spec_num_proposed_drafts']==d*v
 log=(a/'server.log').read_text()
 traces=[x for x in log.splitlines() if 'GEMMA31_SPLITKV_' in x]
 assert any(f"'width': {d+1}," in x and "'dim': 256" in x for x in traces)
 assert any(f"'width': {d+1}," in x and "'dim': 512" in x for x in traces)
 v={'depth':d,'requests':len(rows),'replacement_characters':0,'kernel_traces':traces}
 for lang in ['ko','code']:
  speed=[x for x in rows if x['kind']=='speed_'+lang]
  assert len(speed)==3 and all(x['completion_tokens']==512 for x in speed)
  good=sum(x['meta_info']['spec_num_correct_drafts'] for x in speed)
  proposed=sum(x['meta_info']['spec_num_proposed_drafts'] for x in speed)
  verify=sum(x['meta_info']['spec_verify_ct'] for x in speed)
  v[lang]={'tg_median':statistics.median(x['tg_tok_s'] for x in speed),'tg_values':[x['tg_tok_s'] for x in speed],'accepted':good,'proposed':proposed,'acceptance':good/proposed,'accepted_per_verify':good/verify,'ttft_median_s':statistics.median(x['ttft_s'] for x in speed)}
 summary[str(d)]=v
subprocess.run([sys.executable,'/home/edp1096/.cache/model-download-jobs/gemma31-spec-opt-20261021-183237/evaluate.py',*[str(r/f'splitkv-b4-high-{d}') for d in depths],'--out',str(r/'quality.json')],check=True)
quality=json.loads((r/'quality.json').read_text());assert all(x['passed']==x['total']==2 for x in quality.values())
(r/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
out=Path('/home/edp1096/workspace/dgx_spark_init/util/talk/docs/gemma31-high-mtp-20261021')
out.mkdir(exist_ok=True);(out/'raw').mkdir(exist_ok=True);(out/'kernel-checks').mkdir(exist_ok=True);(out/'harness').mkdir(exist_ok=True)
for n in ['cases.json','summary.json','quality.json','provenance.json','memory-clock.jsonl','selection.json']:shutil.copyfile(r/n,out/n)
for d in depths:
 arm=f'splitkv-b4-high-{d}'
 for n in ['results.json','command.json']:shutil.copyfile(r/arm/n,out/'raw'/f'{arm}-{n}')
for p in (r/'verify-work').glob('*.py'):shutil.copyfile(p,out/'kernel-checks'/p.name)
shutil.copyfile(r/'verify-work/high-width-results.json',out/'kernel-checks/high-width-results.json')
for n in ['run-all.py','run-high-mtp.py','finalize.py']:shutil.copyfile(r/n,out/'harness'/n)
base=Path(json.loads((r/'provenance.json').read_text())['base_job'])
diff=''
for b,n,p in [('triton_backend.py','triton_backend_sm121.py','python/sglang/srt/layers/attention/triton_backend.py'),('verify_splitkv.py','verify_splitkv_sm121.py','python/sglang/kernels/ops/attention/verify_splitkv.py')]:
 diff+=''.join(difflib.unified_diff((base/'source-baseline'/b).read_text().splitlines(True),(r/'verify-work'/n).read_text().splitlines(True),fromfile='a/'+p,tofile='b/'+p))
(out/'runtime.patch').write_text(diff)
lines=['# Gemma 4 31B: higher MTP depths','','One DGX Spark, EXL3 4-bit assistant body / 6-bit head, same target and Split-KV policy as the [main comparison](../gemma31-spec-opt-20261021). Selected: fixed MTP9; MTP15 extension cancelled. Short prompts only; median TG of three matched 512-token samples per language.','','| MTP | Korean TG | Code TG | Korean acceptance | Code acceptance |','| --- | ---: | ---: | ---: | ---: |']
for d in depths:
 v=summary[str(d)];lines.append(f'| {d} | {v["ko"]["tg_median"]:.2f} | {v["code"]["tg_median"]:.2f} | {v["ko"]["acceptance"]:.1%} | {v["code"]["acceptance"]:.1%} |')
lines+=['','All 8 arithmetic/JSON smoke checks passed; no replacement characters. These checks do not establish general quality. New verify widths 19/22/25 passed FP32-reference, pinned-baseline and CUDA-graph checks before serving.','', 'Source: [measurements](summary.json), [raw requests](raw), [settings and base experiment](provenance.json), [kernel qualification](kernel-checks/high-width-results.json).','']
shutil.copytree(r/'runtime-source',out/'runtime-source',dirs_exist_ok=True)
(out/'README.md').write_text('\n'.join(lines))
files=[p for p in sorted(out.rglob('*')) if p.is_file() and p.name!='SHA256SUMS']
(out/'SHA256SUMS').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(out))+'\n' for p in files))
print(json.dumps({k:{lang:round(v[lang]['tg_median'],3) for lang in ['ko','code']} for k,v in summary.items()},indent=2))
print('REPORT',out)
