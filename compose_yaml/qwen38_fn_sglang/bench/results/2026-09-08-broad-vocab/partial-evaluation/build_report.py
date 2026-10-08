"""Reproduce partial-run metrics from preserved, completed response records."""
import json,sys,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parents[2]))
from summarize_broad_vocab import aggregate,geomean,coverage,expand

def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def answered(r):return bool(r['content'].strip()) and (not r['thinking'] or '</think>' in r['content'] and bool(r['content'].split('</think>',1)[1].replace('<|im_end|>','').strip()))
def quality(rs):return dict(requests=len(rs),points=sum(r['score'] for r in rs),max_points=len(rs)*3,full_pass=sum(r['score']==3 for r in rs),material_errors=sum(r['material_error'] for r in rs))
rows={m:read(P/f'{m}-benchmark.json')['rows'] for m in ['ko64k','ko128k']}
assert [len(rows[m]) for m in rows]==[48,18]
q=read(P/'quality-summary.json')['rows']
assert len(q)==66 and len(read(P/'extended/quality-summary.json')['rows'])==8
maps={m:{(r['id'],r['repeat']):r for r in rs} for m,rs in rows.items()}
keys=sorted(set(maps['ko64k']) & set(maps['ko128k']))
assert len(keys)==18 and all(k[1]==0 for k in keys)
sets={m:expand(P.parent.parent/'2026-09-08-vocab-memory'/f'{m}.json') for m in rows}
assert sets['ko64k'] <= sets['ko128k']
close=read(P.parent/'token-metadata.json')['special_token_ids']['</think>']
for rs in rows.values():
 for r in rs:assert len(r['output_ids'])==r['meta_info']['completion_tokens']

def summary(rs,qs):
 ids=[i for r in rs for i in r['output_ids']]
 final=[i for r in rs for i in (r['output_ids'][r['output_ids'].index(close)+1:] if r['thinking'] and close in r['output_ids'] else [] if r['thinking'] else r['output_ids'])]
 return dict(**aggregate(rs),quality=quality(qs),missing_final=sum(not answered(r) for r in rs),coverage=coverage(ids,sets),final_coverage=coverage(final,sets))
paired={m:summary([maps[m][k] for k in keys],[r for r in q if r['mode']==m and (r['id'],r['repeat']) in keys]) for m in rows}
allprimary={m:summary(rs,[r for r in q if r['mode']==m]) for m,rs in rows.items()}
pairs=[]
for k in keys:
 a,b=[maps[m][k] for m in rows]
 grades={m:next(r for r in q if r['mode']==m and (r['id'],r['repeat'])==k) for m in rows}
 pairs.append(dict(id=k[0],repeat=k[1],domain=a['domain'],ratio_128_over_64=b['decode_tok_s']/a['decode_tok_s'],both_finished=all(not r['truncated'] and answered(r) for r in [a,b]),exact_output_match=a['output_ids']==b['output_ids'],grades=grades))
finished=[r['ratio_128_over_64'] for r in pairs if r['both_finished']]
supprows=read(P/'extended/ko64k-benchmark.json')['rows'];suppq=read(P/'extended/quality-summary.json')['rows']
result=dict(status='partial; benchmark remains suspended',completed_responses=74,paired_requests_per_mode=18,paired=paired,all_primary_descriptive_only=allprimary,supplement_64_only=summary(supprows,suppq),pairs=pairs,geomean_ratio_128_over_64=geomean([r['ratio_128_over_64'] for r in pairs]),finished_pairs=len(finished),finished_pair_geomean_ratio_128_over_64=geomean(finished),exact_matches=sum(r['exact_output_match'] for r in pairs))
write(P/'partial-summary.json',result)
lines=['# 재부팅 전 정상 응답의 부분 평가','',
'64k/128k는 컨텍스트 길이가 아닌 draft 후보 어휘 크기다. 두 설정의 컨텍스트와 KV 용량은 65,536으로 고정했다.',
'', '## 평가 범위','',
'- 본 실험 64k 48건, 128k 18건, 64k 추가 실험 8건: 총 74건. 모든 응답 JSON 파싱, 요청 키 중복, 토큰 ID 수를 검증했다.',
'- 재부팅 중 생성하던 128k 19번째 응답은 저장 완료되지 않아 제외했다. 메모리/온도 로그의 손상 꼬리 부분은 답변 점수와 속도 집계에 사용하지 않았다.',
'- 토큰 한도 도달과 최종 답변 부재는 정상 수집된 실패 결과이므로 포함했다. 추론만 있는 요청에는 제출 답변 점수를 주지 않았다.',
'- 설정 비교는 동일 문항·동일 반복 번호 18쌍만 사용했다. 64k의 나머지 30건과 추가 8건을 128k와 직접 비교하지 않았다.',
'- 원본 파일은 보존했다. 이 디렉터리의 benchmark.json은 평가용 부분 자료이며 전체 실험 완료를 뜻하지 않는다.',
'', '## 공통 18쌍 비교','',
'| 항목 | 64k | 128k |','|---|---:|---:|']
for label,fn in [('루브릭 점수',lambda x:f"{x['quality']['points']}/{x['quality']['max_points']}"),('3개 기준 모두 통과',lambda x:f"{x['quality']['full_pass']}/18"),('중요 오류 표시',lambda x:str(x['quality']['material_errors'])),('출력 한도 도달',lambda x:str(x['truncated'])),('최종 답변 없음',lambda x:str(x['missing_final'])),('합산 decode tok/s',lambda x:f"{x['pooled_decode_tok_s']:.2f}"),('합산 전체 응답 tok/s',lambda x:f"{x['pooled_end_to_end_tok_s']:.2f}"),('첫 토큰 지연 중앙값(초)',lambda x:f"{x['median_ttft_s']:.3f}"),('draft 수락률',lambda x:f"{100*x['spec_accept_rate']:.2f}%")]:
 lines.append('| '+label+' | '+' | '.join(fn(paired[m]) for m in rows)+' |')
lines+=['',f"문항별 decode 속도비(128k/64k)의 기하평균은 **{result['geomean_ratio_128_over_64']:.4f}**, 즉 이 표본에서 128k가 약 {(1-result['geomean_ratio_128_over_64'])*100:.1f}% 느렸다. 양쪽 모두 출력 한도에 걸리지 않고 최종 답변을 낸 {len(finished)}쌍만 보면 비율은 {geomean(finished):.4f}이다.",f"출력 토큰열이 완전히 같은 쌍은 {result['exact_matches']}/18개다. 다른 답변의 길이와 난이도가 속도에 영향을 주므로 속도 차이를 어휘 크기의 순수 효과로 단정할 수 없다.",'','점수 차이는 1점이고 중요 오류 수는 반대 방향이다. 품질 우열을 선언할 근거는 부족하다. 중요 오류 표시는 루브릭 점수와 별도이며 세 기준을 통과해도 추가 문장의 오류가 있을 수 있다.','','## 문항별 결과','','| 문항 | 64k 점수/3 | 128k 점수/3 | 속도비 128k/64k |','|---|---:|---:|---:|']
for r in pairs:lines.append(f"| {r['id']} | {r['grades']['ko64k']['score']} | {r['grades']['ko128k']['score']} | {r['ratio_128_over_64']:.3f} |")
lines+=['','## 어휘 범위','','각 모드가 실제 생성한 토큰 ID를 두 후보 집합 모두와 대조했다. 이는 정답률이나 문맥 이해력 지표가 아니다.','','| 생성 모드 | 생성 토큰 | 64k 집합 밖 | 128k 집합 밖 |','|---|---:|---:|---:|']
for m in rows:
 c=paired[m]['coverage'];lines.append(f"| {m} | {c['tokens']} | {c['outside_pct']['ko64k']:.3f}% | {c['outside_pct']['ko128k']:.3f}% |")
lines+=['','## 나머지 응답의 기술적 집계','','64k 본 실험 전체는 99/144점, 3개 기준 통과 24/48건이다. 표본 구성이 달라 128k의 37/54점과 백분율만으로 비교하면 안 된다.',
'','64k 추가 실험은 기존 한도 도달/최종 답변 부재 문항 8개에 한도를 늘린 선택 표본이다. 7/24점, 전체 기준 통과 1/8건이다. Python 코드 2건과 메모리 분석은 늘린 한도에서도 최종 답변이 없었다. 128k 추가 실험은 실행되지 않았다.',
'', '## 주요 오류와 해석','',
'- Python 두 문항은 양쪽 모두 최종 코드 미제출로 0점이다. 코드 정확도가 실행 테스트에서 0%였다는 의미가 아니다. 자동 채점기는 제출 여부 검사에서 종료했다.',
'- HTML 리뷰는 숨긴 링크·표 처리의 일부 문제를 찾았지만 재현/회귀 테스트와 공개 콘텐츠 유지에 오류가 있었다. 추가 토큰 실험은 오히려 공개 링크 텍스트를 제거하는 잘못된 수정으로 진행했다.',
'- 의학 답변은 긴 설명이 있어도 긴급 대응 조건 누락이나 부적절한 지연 권고 등이 있었으며, 식품 보관 답변도 중요한 안전 판단을 놓쳤다. 항목별 근거는 quality-summary.json의 notes에 보존했다.',
'- 일정은 기본 시간표가 명시된 체류시간/폐장 조건을 만족하지 못했다. 추가 실험은 공원 시간을 60분에서 45분으로 줄였고, 식사 이후 휴식을 공원 방문 뒤에 배치할 수 있다는 선택을 놓쳤다.',
'- 번역·업무 공지·대화의 최신 예산 반영은 64k의 해당 6건 모두 루브릭을 통과했다. 해당 문항의 128k 결과는 없다.',
'', '## 한계와 판단','',
'정상 응답만으로 판단하면 128k의 품질 향상은 입증되지 않았고, 공통 18개 문항의 속도는 64k가 우세했다. 이는 64k를 유지할 잠정 근거이며 범용 성능의 확정 결론은 아니다.',
'',
'128k는 첫 반복의 앞 18문항까지만 남았다. 모델 실행 순서, 온도, 64k 중간 재시작, 백그라운드 상태 조회, 설명 길이 차이가 통제되지 않았다. 재부팅 원인은 불명이며 이번 평가는 충돌 원인을 규명하지 않는다.',
'',
'각 문항 3개의 고정 기준으로 평가했다. 서술형은 설정 이름을 숨겨 작성한 기존 평가 48개를 재사용하고, 새 답변도 같은 기준으로 검토했다. 다만 검토자는 동일 assistant이며 독립 전문가/완전한 맹검 평가는 아니다. 확대된 표본의 의료·법률 전문 검증이나 실제 서비스 신뢰성 인증이 아니다.',
'', '## 상세 자료','',
'- [기계 판독 집계](partial-summary.json)',
'- [본 실험 개별 점수와 판정 이유](quality-summary.json)',
'- [추가 실험 개별 점수와 판정 이유](extended/quality-summary.json)',
'- [64k 원문 응답](ko64k-benchmark.json) / [128k 원문 응답](ko128k-benchmark.json)',
'- [64k 추가 원문](extended/ko64k-benchmark.json)',
'- [원래 사전 계획](../PLAN.md)',
'- [재현 스크립트](build_report.py)',
'']
(P/'README.md').write_text('\n'.join(lines))
write(P/'evaluation-hashes.json',{str(f.relative_to(P)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [P/'build_report.py',P/'tasks.json',P/'prose-reviews.json',P/'code-grades.json',P/'ledger-grades.json',P/'quality-summary.json',P/'extended/prose-reviews.json',P/'extended/quality-summary.json']})
print(json.dumps({k:v for k,v in result.items() if k in ['geomean_ratio_128_over_64','finished_pairs','finished_pair_geomean_ratio_128_over_64','exact_matches']},indent=2))
