#!/usr/bin/env python3
"""Write a Korean summary and a standalone interactive response comparison."""
import argparse
import json
from pathlib import Path

DOMAINS={'medicine':'의학','everyday':'생활·문제 해결','planning':'일정·대화 조건','economics':'가계·비용 계산','documents':'계약·원장 독해','science':'과학·수리','research':'연구·기술 분석','coding':'코딩','language':'번역·글쓰기','humanities':'인문·사료 독해'}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path);args=ap.parse_args()
    out=args.directory.resolve()
    s=json.loads((out/'summary.json').read_text());q=json.loads((out/'quality-summary.json').read_text())
    tasks=json.loads((out/'tasks.json').read_text())['tasks']
    a,b=s['modes']['ko64k'],s['modes']['ko128k']
    qa,qb=q['modes']['ko64k'],q['modes']['ko128k']
    def pct(r):return f'{(r-1)*100:+.2f}%'
    lines=['# 범용 작업에서 draft vocabulary 64K / 128K 비교','',
        '10개 분야의 고정된 24개 과제를 설정별 2회, 총 96회 실행했습니다. '
        '네 과제는 추론 모드이며, 나머지는 추론을 끈 답변입니다. '
        '실사용 빈도를 추정한 표본이 아니라 분야를 넓힌 통제 실험입니다.','',
        f'**128K의 과제 균등 디코딩 속도 변화: {pct(s["task_balanced_decode_ratio_128_over_64"])}.** '
        f'분야 균등 변화는 {pct(s["domain_balanced_decode_ratio_128_over_64"])}입니다. '
        f'5% 넘게 빠른 과제는 {s["tasks_128_faster_over_5pct"]}/24개, '
        f'5% 넘게 느린 과제는 {s["tasks_128_slower_over_5pct"]}/24개입니다.','',
        '| 지표 | 64K | 128K |','|---|---:|---:|',
        f'| 합산 디코딩 처리량 (tok/s) | {a["pooled_decode_tok_s"]:.2f} | {b["pooled_decode_tok_s"]:.2f} |',
        f'| 합산 요청 처리량, 입력 처리 포함 (tok/s) | {a["pooled_end_to_end_tok_s"]:.2f} | {b["pooled_end_to_end_tok_s"]:.2f} |',
        f'| 요청 실행시간 합계 (분) | {a["elapsed_s"]/60:.1f} | {b["elapsed_s"]/60:.1f} |',
        f'| 생성 토큰 합계, 추론 포함 | {a["tokens"]:,} | {b["tokens"]:,} |',
        f'| 초안 채택률 | {a["spec_accept_rate"]*100:.2f}% | {b["spec_accept_rate"]*100:.2f}% |',
        f'| 검증 1회당 확정 토큰, 보너스 포함 | {a["spec_accept_length"]:.3f} | {b["spec_accept_length"]:.3f} |',
        f'| 기준 충족 점수 | {qa["points"]}/{qa["max_points"]} | {qb["points"]}/{qb["max_points"]} |',
        f'| 3개 기준 모두 충족한 응답 | {qa["full_pass_requests"]}/48 | {qb["full_pass_requests"]}/48 |',
        f'| 출력 한도 도달 | {a["truncated"]} | {b["truncated"]} |',
        f'| 추론만 있고 최종 답변 없음 | {a["missing_answers"]} | {b["missing_answers"]} |',
        f'| 종료 후 대기 메모리 증가 (GiB) | {a["post_idle_extra_memory_gib"]:.2f} | {b["post_idle_extra_memory_gib"]:.2f} |',
        f'| 전체 측정 최고 메모리 증가 (GiB) | {a["peak_extra_memory_gib"]:.2f} | {b["peak_extra_memory_gib"]:.2f} |',
        f'| 가용 메모리 최저 (GiB) | {a["min_available_gib"]:.2f} | {b["min_available_gib"]:.2f} |','',
        '메모리는 모델을 띄우기 전 가용 메모리 대비 증가량입니다. '
        '요청 실행시간에는 모델 로딩·캐시 비우기·대기 시간이 포함되지 않습니다. '
        '출력 길이가 다를 수 있어 완료 시간과 토큰 처리 속도는 별도로 해석해야 합니다.','',
        '## 분야별 결과','',
        '| 분야 | 과제 수 | 128K 속도 변화¹ | 64K 기준 점수 | 128K 기준 점수 |','|---|---:|---:|---:|---:|']
    for d in s['per_domain']:
        name=d['domain'];x=qa['domains'][name];y=qb['domains'][name]
        lines.append(f'| {DOMAINS[name]} | {d["tasks"]} | {pct(d["task_balanced_decode_ratio_128_over_64"])} | {x["points"]}/{x["max_points"]} | {y["points"]}/{y["max_points"]} |')
    complete_ratio=s.get('complete_pair_task_balanced_decode_ratio_128_over_64')
    if complete_ratio is not None:
        lines+=['',f'두 설정 모두 출력 한도에 도달하지 않고 최종 답변을 생성한 {s["complete_pairs"]}/48쌍만의 '
                f'보조 집계에서는, 과제 균등 속도 변화가 **{pct(complete_ratio)}**였습니다. '
                '이 집계는 잘림 관측 후 추가한 민감도 확인이며, 포함 과제 구성이 전체 집계와 다릅니다.']
    lines+=['','¹ 양수는 128K가 빠름. 각 과제의 반복별 속도 비율을 기하평균했습니다. '
        '과제별 표본이 작고 모든 64K 실행 후 128K를 실행하므로 작은 차이를 확정적 우열로 보지 않습니다.','',
        '## 실제 생성 어휘','',
        '| 생성 설정 | 생성 토큰 중 64K 밖 | 생성 토큰 중 128K 밖 | 128K가 추가로 포함한 토큰 |','|---|---:|---:|---:|']
    for mode,label in [('ko64k','64K'),('ko128k','128K')]:
        c=s['modes'][mode]['coverage']
        lines.append(f'| {label} | {c["outside_pct"]["ko64k"]:.3f}% | {c["outside_pct"]["ko128k"]:.3f}% | {c["recovered_by_128k"]:,} |')
    lines+=['','실제 출력 토큰 ID를 사용했으며, 추론 및 종료 특수 토큰을 포함합니다. '
        '최종 답변만의 비율은 [summary.json](summary.json)에 별도로 기록했습니다. '
        '64K 밖 토큰도 본 모델은 생성할 수 있습니다. 위 비율 자체가 초안 채택률은 아닙니다.','',
        '## 과제별 속도','',
        '| 과제 | 64K tok/s² | 128K tok/s² | 128K 변화¹ | 반복별 변화 |','|---|---:|---:|---:|---|']
    for t in s['per_task']:
        x=t['modes']['ko64k'];y=t['modes']['ko128k']
        repeats=', '.join(pct(r) for r in t['repeat_decode_ratios'])
        lines.append(f'| {t["id"]} | {x["pooled_decode_tok_s"]:.2f} | {y["pooled_decode_tok_s"]:.2f} | {pct(t["decode_ratio_128_over_64"])} | {repeats} |')
    lines+=['','² 설정별 두 응답을 합산한 처리량이므로, 비율의 기하평균인 변화 열과 정확히 일치하지 않을 수 있습니다.','',
        '## 평가 방법과 한계','',
        '- 각 과제의 세 가지 기준을 실행 전에 고정했습니다. 코딩 두 과제에는 총 27개 기능 테스트를 준비했으며, 최종 코드가 제출된 응답만 실행 검사합니다. 최종 코드가 없는 응답은 미제출로 구분합니다. 원장은 정확한 집계값과 비교했습니다.',
        '- 서술 답변은 설정 이름을 가리고 조수가 판정했습니다. 동일한 최종 답변은 중복 판정을 제거했습니다. 독립 전문가 심사나 의학·법률 분야의 포괄적 정확도 인증은 아닙니다.',
        '- 문맥 길이와 KV 풀은 모두 65,536, Mamba 슬롯 18, MTP steps 3 / draft tokens 4로 고정했습니다. 운영 환경의 자동 KV 크기와 다릅니다.',
        '- temperature 0, 동일한 채팅 템플릿, 동일한 과제별 출력 한도입니다. 매 요청 전 prefix cache를 비웠고, 준비용 요청은 통계에서 제외했습니다.',
        '- 디코딩 속도는 첫 스트림 묶음 이후 생성 토큰 수를 마지막 새 토큰까지의 시간으로 나눴습니다. 첫 묶음에 여러 토큰이 들어오는 경우를 반영했습니다.',
        '- 모든 64K 실행 후 128K를 실행했습니다. 두 번째 반복은 과제 순서를 뒤집었지만, 모델 실행 순서 및 장비 상태에 따른 차이는 완전히 제거하지 못합니다.',
        '- 64K의 첫 16개 완료 응답 뒤, CPU 보조 컨테이너를 과도하게 제한한 감시 조건 때문에 한 번 중단됐습니다. 미완료 요청을 버리고 같은 이미지·설정으로 재로딩해 이어갔습니다. [중단 기록](interruption.json)과 원래 로그를 보존했습니다.',
        '- 재개 후에는 확인된 SSH 키 저장소의 CPU 상태 조회 컨테이너만 허용했습니다. GPU 작업 및 메모리·swap 중단 조건은 유지했습니다. 따라서 작은 속도 차이는 재로딩과 배경 활동의 영향도 고려해야 합니다.',
        '- 긴 문서·대화는 고정한 입력을 재생했습니다. 실제 브라우저/도구 호출, 장기 자율 에이전트, 이미지·음성·동시 사용은 이번 비교에 포함하지 않았습니다.',
        '- 전체 작업 빈도를 알 수 없으므로 개인의 실사용 평균이나 모든 분야의 우열로 일반화하지 않습니다.',
        f'- OOM 없이 완료했습니다. Swap 사용 최대 증가량은 64K {a["max_swap_growth_mib"]:.1f} MiB, 128K {b["max_swap_growth_mib"]:.1f} MiB였습니다. 운영 모델은 종료 상태, 지원 서비스는 기존 상태를 유지했습니다.','',
        '## 결과 확인','',
        '- [응답·점수·속도 비교 화면](comparison.html)',
        '- [사전 계획](PLAN.md) · [고정 과제와 근거](tasks.json)',
        '- [속도·어휘 집계](summary.json) · [품질 판정과 근거](quality-summary.json)',
        '- [코드 실행 결과](code-grades.json) · [원장 계산 결과](ledger-grades.json)',
        '- [64K 원시 응답](ko64k-benchmark.json) · [128K 원시 응답](ko128k-benchmark.json)',
        '- [실행 조건](run.json) · [완료 및 상태 보존](complete.json)','',
        '- [출력 한도를 늘린 별도 보충 검증](extended/README.md)','',
        '의학 판정에 사용한 자료:']
    seen=set()
    for t in tasks:
        if t['domain']=='medicine':
            for source in t['sources']:
                if source['url'] not in seen:
                    seen.add(source['url']);lines.append(f'- [{t["id"]}]({source["url"]})')
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    rows=json.loads((out/'rows-with-coverage.json').read_text())
    payload=json.dumps(dict(summary=s,quality=q,tasks=tasks,rows=rows,domains=DOMAINS),ensure_ascii=False).replace('<','\\u003c')
    html=r'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>64K / 128K 범용 작업 비교</title>
<style>body{font-family:system-ui,sans-serif;background:#f5f6f8;color:#17212d;margin:0;padding:24px;max-width:1500px;margin:auto}h1{font-size:26px}p{line-height:1.6}.controls{display:flex;gap:12px;flex-wrap:wrap;margin:20px 0}select{padding:10px;font-size:16px;max-width:100%}.cards{display:grid;grid-template-columns:1fr 1fr;gap:16px}.card,details{background:white;border:1px solid #dce0e5;border-radius:10px;padding:18px;margin:12px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.65 ui-monospace,monospace}.metrics{color:#31506b;line-height:1.8}.badge{font-size:14px;color:#48566b}summary{cursor:pointer}a{color:#155ea2}@media(max-width:850px){.cards{grid-template-columns:1fr}body{padding:12px}}</style>
<h1>64K / 128K 범용 작업 비교</h1><p id="overall"></p><p><a href="README.md">전체 보고서</a> · 같은 과제의 응답, 고정 기준 점수와 측정값을 비교합니다. 서술 평가는 조수의 판정이며 독립 전문가 심사가 아닙니다.</p>
<div class="controls"><select id="domain"><option value="">모든 분야</option></select><select id="task"></select><select id="repeat"><option value="0">반복 1</option><option value="1">반복 2</option></select></div>
<details><summary>입력과 평가 기준</summary><pre id="prompt"></pre><pre id="rubric"></pre></details><div class="cards" id="cards"></div>
<script id="data" type="application/json">PAYLOAD</script><script>
const D=JSON.parse(document.getElementById('data').textContent);const $=id=>document.getElementById(id);
$('overall').textContent=`24개 과제 × 2회 × 2설정. 과제 균등 디코딩 속도 변화(128K/64K): ${((D.summary.task_balanced_decode_ratio_128_over_64-1)*100).toFixed(2)}%. 양수는 128K가 빠릅니다.`;
for(const [k,v] of Object.entries(D.domains)){const o=new Option(v,k);$('domain').add(o)}
function setTasks(){const selected=$('task').value;$('task').replaceChildren();for(const t of D.tasks){if(!$('domain').value||t.domain===$('domain').value)$('task').add(new Option(t.id,t.id))}if([...$('task').options].some(o=>o.value===selected))$('task').value=selected;render()}
function pre(parent,text){const p=document.createElement('pre');p.textContent=text;parent.append(p)}
function render(){const t=D.tasks.find(t=>t.id===$('task').value);if(!t)return;const repeat=Number($('repeat').value);$('prompt').textContent=t.messages.map(m=>m.role+':\n'+m.content).join('\n\n');$('rubric').textContent=t.rubric.map((r,i)=>(i+1)+'. '+r).join('\n');$('cards').replaceChildren();for(const mode of ['ko64k','ko128k']){const r=D.rows[mode].find(r=>r.id===t.id&&r.repeat===repeat);const g=D.quality.rows.find(r=>r.mode===mode&&r.id===t.id&&r.repeat===repeat);const c=document.createElement('section');c.className='card';const h=document.createElement('h2');h.textContent=mode==='ko64k'?'64K':'128K';c.append(h);const m=document.createElement('p');m.className='metrics';m.textContent=`${r.decode_tok_s.toFixed(2)} tok/s · 완료 ${r.elapsed_s.toFixed(1)}초 · 첫 토큰 ${r.ttft_s.toFixed(2)}초 · ${r.output_ids.length}토큰\n초안 채택 ${(r.meta_info.spec_accept_rate*100).toFixed(2)}% · 기준 ${g.score}/3 · 출력 한도 ${r.truncated?'도달':'미도달'} · 최종 답변 ${r.answer_present?'있음':'없음'}`;c.append(m);pre(c,'평가: '+g.groups.map((v,i)=>(i+1)+':'+(v?'통과':'미충족')).join(' / ')+'\n'+g.notes);let answer=r.content;let thought='';if(r.thinking){const split=answer.indexOf('</think>');if(split>=0){thought=answer.slice(0,split);answer=answer.slice(split+8)}else{thought=answer;answer='[최종 답변 없음]'}}pre(c,answer.replaceAll('<|im_end|>',''));if(thought){const d=document.createElement('details');const sm=document.createElement('summary');sm.textContent='실험 모델의 추론 출력';d.append(sm);pre(d,thought);c.append(d)}$('cards').append(c)}}
$('domain').addEventListener('change',setTasks);$('task').addEventListener('change',render);$('repeat').addEventListener('change',render);setTasks();
</script></html>'''.replace('PAYLOAD',payload)
    (out/'comparison.html').write_text(html)
    print('README.md and comparison.html generated')

if __name__=='__main__':main()
