"""Summarize externally measured fixed-output requests and functional checks."""
import argparse,ast,json,pathlib,statistics
p=argparse.ArgumentParser();p.add_argument('directory',type=pathlib.Path);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
def load(name):return json.loads((a.directory/name).read_text())
summary={'timing':'Wall-clock HTTP completion, exactly 256 output tokens; round 0 excluded; median of 3. No claim of identical generated tokens or MTP acceptance.'}
fixed={label:load(label+'-fixed.json') for label in ('baseline','candidate')}
summary['fixed_output']={}
for name in ('ko','code'):
 stats={}
 for label,rows in fixed.items():
  selected=[r for r in rows if r['case']==name and r['round']>0]
  assert len(selected)==3
  assert all(r['result']['meta_info']['completion_tokens']==256 for r in selected)
  median=statistics.median(r['seconds'] for r in selected)
  stats[label]={'median_seconds':median,'tokens_per_wall_second':256/median,
    'round_seconds':[r['seconds'] for r in selected],
    'accept_lengths':[r['result']['meta_info']['spec_accept_length'] for r in selected]}
 stats['wall_time_reduction_pct']=100*(1-stats['candidate']['median_seconds']/stats['baseline']['median_seconds'])
 summary['fixed_output'][name]=stats
summary['functional']={}
for label in ('baseline','candidate'):
 rows={r['name']:r for r in load(label+'-responses.json')}
 def text(name):return rows[name]['response']['choices'][0]['message']['content'].strip()
 checks={'arithmetic_ko':'323' in text('ko'),'arithmetic_ko_repeat':'323' in text('ko_repeat'),
 'json':json.loads(text('json'))=={'name':'sample','count':7,'enabled':True},
 'long_recall':text('long')=='BLUE-731','prefix_reuse':text('long_reuse')=='BLUE-731','branch':text('branch')=='42'}
 code=text('code')
 if code.startswith('```'):
  code=code.split('\n',1)[1].rsplit('```',1)[0]
 checks['code_syntax']=any(isinstance(n,ast.FunctionDef) and n.name=='binary_search' for n in ast.parse(code).body)
 media={r['name']:r for r in load(label+'-media.json')}
 msg=lambda name:media[name]['response']['choices'][0]['message']
 checks.update(image=msg('image')['content'].strip().lower()=='red',
   video=msg('video')['content'].strip().lower().replace(',','').split()==['red','blue'],
   tool=msg('tool')['tool_calls'][0]['function']['name']=='get_weather' and json.loads(msg('tool')['tool_calls'][0]['function']['arguments'])['city']=='Seoul')
 summary['functional'][label]=checks
 assert all(checks.values()),checks
 summary.setdefault('long_prefix',{})[label]={name:{'seconds':rows[name]['seconds'],'prompt_tokens':rows[name]['response']['usage']['prompt_tokens']} for name in ('long','long_reuse','branch')}
a.output.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');print(json.dumps(summary,ensure_ascii=False,indent=2))
