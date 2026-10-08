#!/usr/bin/env python3
"""Freeze a held-out, mixed-domain corpus and evaluation rubrics before serving."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
OUT = ROOT / 'results/2026-09-08-broad-vocab'
tasks = []

def add(ident, domain, prompt, rubric, *, thinking=False, limit=1536, sources=None, messages=None):
    tasks.append(dict(id=ident, domain=domain, messages=messages or [dict(role='user', content=prompt)],
                      rubric=rubric, thinking=thinking, max_tokens=limit, sources=sources or []))

add('med_cold','medicine',
'''가상의 건강상담입니다. 38세 직장인, 콧물·기침이 3일 있었고 잠깐 나아졌다가 오늘 다시 열이 나며 숨이 차다고 합니다. 집에 남은 항생제를 먹고 출근해도 되는지 묻습니다. 바이러스성 감기와 다른 원인을 단정하지 말고, 지금의 우선 행동, 항생제 판단, 증상 완화와 주변 전파 예방을 한국어로 설명하세요. 기저질환·복용약은 아직 모릅니다.''',
['Breathlessness and relapse trigger prompt medical assessment rather than routine home care only.',
 'Does not recommend leftover antibiotics or claim antibiotics cure viral colds.',
 'Gives supportive care and asks/checks relevant missing information without a definitive diagnosis.'],
 sources=[dict(url='https://www.cdc.gov/common-cold/treatment/index.html',reference='Viral colds do not benefit from antibiotics. Breathing difficulty and improvement followed by deterioration warrant medical care; rest and fluids help symptoms.')])
add('med_gout','medicine',
'''통풍 진단을 받고 요산을 낮추는 약을 복용 중인 사람이 최근 1년 동안 발작을 세 번 겪었습니다. 지금은 통증이 없으니 약을 끊고 맥주만 안 마시면 되는지, 콜히친과 요산저하제가 같은 역할인지 묻습니다. 급성 발작 치료와 장기 예방을 구분하고, 식생활의 역할과 한계, 담당의와 확인할 항목을 설명하세요. 약의 용량과 신장기능 수치는 제공되지 않았습니다. 한국어로 답하세요.''',
['Distinguishes acute anti-inflammatory treatment from sustained urate reduction.',
 'Does not advise independently stopping established urate-lowering medication just because pain resolved.',
 'Explains lifestyle alone can be insufficient with recurrent flares and identifies medication/renal review.'],
 sources=[dict(url='https://www.niams.nih.gov/health-topics/gout/diagnosis-treatment-and-steps-to-take',reference='Flare treatment reduces inflammation and pain. Urate-lowering therapy targets hyperuricemia and recurrence. With frequent flares, lifestyle changes alone are insufficient; medication changes require clinician discussion.')])
add('med_rhinitis','medicine',
'''알레르기 비염이 있는 성인이 코막힘 때문에 혈관수축성 비충혈제거 스프레이를 3주째 매일 사용합니다. 뿌릴 때만 잠깐 낫고 더 자주 막힙니다. 항히스타민제·비강 스테로이드·생리식염수와 어떻게 다른지, 계속 횟수를 늘려도 되는지 물었습니다. 장기 사용 위험과 다음 행동을 설명하되 모든 비강 스프레이를 같은 약처럼 취급하지 마세요. 한국어로 답하세요.''',
['Identifies rebound congestion from prolonged topical decongestant use and does not advise escalation.',
 'Distinguishes decongestants from intranasal steroids, antihistamines and saline.',
 'Recommends pharmacist/clinician review and trigger management rather than declaring all sprays addictive.'],
 sources=[dict(url='https://www.nhs.uk/conditions/allergic-rhinitis/',reference='Allergic rhinitis treatments include antihistamines, steroid sprays and saline. Prolonged nasal decongestant use can worsen congestion. Persistent or disruptive symptoms warrant review.'),dict(url='https://www.nhs.uk/medicines/decongestants/',reference='NHS currently limits decongestant nasal sprays/drops to five days at a time; other medicines and chronic conditions affect suitability.')])
add('med_diabetes','medicine',
'''가상의 응급대응 교육 문제입니다. 인슐린 치료 중인 성인이 식사를 거른 뒤 식은땀과 손떨림이 있고 혈당이 58 mg/dL입니다. 현재는 의식이 명료하고 삼킬 수 있습니다. 보호자에게 즉시 할 일과 재측정 시점을 설명하고, 이후 의식을 잃거나 삼키지 못하게 되면 대응이 어떻게 달라지는지도 구분하세요. 장기 인슐린 용량을 임의로 새로 처방하지 마세요.''',
['For conscious swallowing patient recommends 15–20 g fast carbohydrate/glucose.',
 'Rechecks after approximately 15 minutes and repeats treatment if still low.',
 'If unconscious/cannot swallow avoids oral intake and calls emergency help; mentions glucagon when available.'],
 sources=[dict(url='https://www.niddk.nih.gov/health-information/diabetes/overview/preventing-problems/low-blood-glucose-hypoglycemia',reference='Below 70 mg/dL is low for many people. Treat with 15–20 g carbohydrate and recheck after 15 minutes. Severe hypoglycemia may prevent self-treatment and requires help/glucagon.')])
add('med_cancer','medicine',
'''건강검진에서 종양표지자 하나가 기준보다 약간 높게 나온 45세 사람이 "이제 암 확정인가요? 정상인 다른 표지자들은 암이 없다는 뜻인가요? 전신 검사를 매달 해야 하나요?"라고 묻습니다. 종양표지자의 비특이성·민감도 한계, 영상·조직검사 및 병력과의 관계, 다음 진료에서 확인할 질문을 불안을 키우지 않게 설명하세요. 구체적인 표지자 이름과 수치는 아직 모릅니다.''',
['Elevated marker alone does not establish cancer and benign causes are possible.',
 'Normal markers cannot exclude all cancers; diagnosis integrates clinical and other test evidence.',
 'Asks marker/value/history and recommends contextual follow-up without inventing diagnosis or monthly whole-body screening.'],
 sources=[dict(url='https://www.cancer.gov/about-cancer/diagnosis-staging/diagnosis/tumor-markers-fact-sheet',reference='Some benign conditions elevate markers and some cancers do not. Markers are combined with imaging/biopsy and other findings; many are unsuitable for population screening.')])

add('life_food','everyday',
'''도시락 준비 기록: A는 닭고기와 밥을 조리한 뒤 실온 24도에서 4시간 두었다가 냉장했습니다. B는 조리 후 30분 안에 냉장했고 냉장고는 4도, 오늘이 2일째입니다. A도 냄새가 괜찮고 전자레인지로 뜨겁게 데우면 먹을 수 있다는 가족을 설득하는 짧은 답변을 작성하세요. A/B를 구분하고 다음 주 준비 습관도 제안하세요.''',
['Discards A rather than declaring reheating or smell sufficient.', 'Treats promptly refrigerated B differently, with appropriate storage/handling qualifications.', 'Explains time/temperature control and practical prompt refrigeration.'],
 sources=[dict(url='https://www.fsis.usda.gov/food-safety/safe-food-handling-and-preparation/food-safety-basics/leftovers-and-food-safety',reference='Perishable leftovers left at room temperature over two hours should be discarded. Properly refrigerated leftovers generally keep three to four days.')])
add('life_energy','everyday',
'''실제 제품 추천이 아닌 주어진 수치만의 가계 계산입니다. 기존 제습기는 평균 400 W로 하루 6시간, 새 제품은 250 W로 같은 시간 작동합니다. 연 120일 사용, 전기요금은 누진제 없이 1 kWh당 250원으로 고정합니다. 새 제품은 180,000원이며 기존 제품은 정상입니다. 연간 절약 전력·금액과 단순 회수기간을 계산하고 소음·성능이 동등하다는 가정 및 실제와 다른 점을 설명하세요.''',
['Calculates annual saving 108 kWh.', 'Calculates 27,000 KRW/year and payback approximately 6.67 years.', 'Separates fixed-rate/equal-performance assumptions from real appliance decisions.'])
add('life_router','everyday',
'''집 인터넷 장애 기록입니다. 노트북에서 게이트웨이 192.168.1.1 ping 성공, 1.1.1.1 ping 성공, example.com 이름 조회는 timeout입니다. 같은 공유기의 휴대전화는 Wi-Fi에서 실패하고 모바일 데이터에서는 성공합니다. 오늘 공유기에 DNS 필터를 추가했습니다. "공장초기화부터 하자"는 제안 대신 원인을 좁히는 순서를 써 주세요. 명령 예시와 예상 결과, 되돌릴 변경, ping 성공이 모든 연결 정상임을 뜻하지 않는 이유를 포함하세요.''',
['Prioritizes DNS/filter configuration based on evidence.', 'Proposes discriminating lookup tests and reversible rollback, not reset-first.', 'Does not infer all TCP/HTTPS paths healthy from ICMP alone.'])
add('life_schedule','planning',
'''외부 검색 없이 다음 가상 시설 정보만 사용하세요. 가족 일정은 10:00~16:30, 시작과 끝은 역 S. S→박물관 M 20분, M→식당 R 10분, R→공원 P 15분, P→S 25분. M 체류 90분, 입장 가능 10:30~12:00. R 예약 12:30, 식사 60분. P 체류 60분. 모든 이동은 같은 시간이 걸리는 택시이며 어르신을 위해 R 이후 쉬는 시간 30분을 확보해야 합니다. 공원은 15:00 폐장입니다. 시간표를 만들고 대기·휴식을 표시하고 30분 지연 시 대안까지 설명하세요. 기재하지 않은 시설이나 이동 시간을 만들어 넣지 마세요.''',
['Museum admission and 12:30 meal reservation both respected.', 'Includes 30-minute post-meal rest and finishes park by 15:00 with return by 16:30.', 'Delay alternative explicitly adjusts feasible activities without invented travel facts.'])

add('finance_budget','economics',
'''아래는 가상 가계 데이터이며 특정 금융상품 추천이 아닙니다. 월 실수령 320만원, 월 고정비 150만원, 변동비 80만원, 최소 부채상환 20만원. 현재 비상금 100만원, 추가 이자·세금은 계산에서 제외합니다. 월 잉여금 전부를 비상금에 넣을 때 필수지출(고정비+변동비+최소상환)의 3개월분까지 얼마나 부족하고 몇 개월이 필요한지 계산하세요. 소득이 20% 줄어들면 같은 목표 도달기간이 어떻게 바뀌는지도 계산하고 가정의 한계를 설명하세요.''',
['Baseline surplus 70만원, target 750만원 and shortfall 650만원.', 'Rounds baseline months up to 10.', 'Reduced income 256만원, surplus 6만원 and 109 months; discusses assumptions.'])
add('finance_cost','economics',
'''가상 장비 구입 계약 두 개를 비교하세요. 현금가격 1,200,000원. A는 선납금 200,000원과 매월 95,000원씩 12회, 다른 비용 없음. B는 선납금 없이 매월 105,000원씩 12회와 최초 수수료 50,000원. 할인율·세금 없이 총 명목 지출을 비교하고, 현금 대비 추가 비용과 A/B 차이를 계산하세요. 이 숫자만으로 APR을 단순 추가비용/현금가격이라고 단정할 수 있는지도 설명하세요.''',
['A total 1,340,000 and B total 1,310,000 KRW.', 'Extra cash costs 140,000/110,000 and B cheaper by 30,000.', 'Does not equate simple markup to APR; cash flow timing matters.'],
 sources=[dict(url='https://www.consumerfinance.gov/ask-cfpb/what-is-an-annual-percentage-rate-apr-and-why-is-it-higher-than-the-interest-rate-for-my-payday-loan-en-1625/',reference='APR expresses borrowing cost on an annualized basis and depends on fees and the borrowing period.')])
add('contract_reading','documents',
'''실제 법률 자문이 아닌 가상 계약 독해 과제입니다. 적용 조항: [1] 월 이용료 50,000원, 매월 1일 선납. [2] 해지 통보는 해당 월 말 7일 전까지 서면 접수해야 다음 달 청구가 없다. [3] 이미 개시한 월의 중도해지 환불은 없다. [4] 사업자 귀책으로 연속 72시간 이상 서비스 중단 시 해당 월 이용료의 50%를 환급한다. [5] [4]의 환급은 [3]보다 우선하며 별도 청구가 필요하다. 30일로 끝나는 4월에 고객이 4월 20일 이메일로 해지를 접수했고, 같은 달 사업자 귀책 중단이 연속 80시간 있었다. 5월 청구와 4월 환급을 구분하고 금액·근거 조항을 제시하세요. 실제 강행법규의 효력 판단은 자료만으로 할 수 없음을 구분하세요.''',
['No May charge because notice was timely.', 'April refund 25,000 based on clauses 4–5 despite clause 3.', 'Separates fictional contractual reading from real-law enforceability.'])

add('science_bayes','science',
'''가상의 공장 검사 문제입니다. 부품 불량률 1%, 검사 민감도 90%, 특이도 95%입니다. 10,000개를 검사한다고 가정하여 참양성·거짓양성을 계산하고, 양성 부품이 실제 불량일 확률을 구하세요. "검사가 90% 정확하므로 양성이면 90% 불량"이라는 주장을 교정하고, 동일한 검사 두 번의 결과가 조건부 독립인지 모르면 확률을 함부로 제곱할 수 없는 이유를 설명하세요.''',
['Computes TP 90 and FP 495.', 'Posterior 90/585 approximately 15.38%.', 'Explains base rates and conditional-dependence issue.'])
add('science_trial','research',
'''다음은 실제 논문이 아닌 가상 연구 기록입니다. 연구 A: 무작위 배정, 치료군 사건 20/200, 대조군 40/200, 추적 12개월, 두 군 탈락 각각 5%, 사전등록된 1차 평가변수. 연구 B: 자발적 앱 사용자 40/800 대 비사용자 90/900, 사용자 평균 연령이 12세 낮고 운동량이 더 많음, 보정 전 관찰자료. 연구 C: 15명의 단일군 전후 설문, 20개 지표 중 1개만 p<0.05, 사전등록 없음. 세 연구를 종합하는 한국어 검토 메모를 쓰세요. A의 ARR·RR·RRR·NNT를 계산하고 A/B 인과근거 차이, C의 다중비교·선택보고 문제, 추가로 필요한 자료를 구분하세요. 존재하지 않는 논문명이나 신뢰구간을 만들지 마세요.''',
['A ARR .10, RR .5, RRR .5 and NNT 10 at 12 months.', 'B association is confounded; A randomization is stronger causal evidence with limitations.', 'C small uncontrolled multiple-testing/selective-reporting limitations; no fabricated CI/publication.'], thinking=True,limit=4096)
add('science_lab','science',
'''실험 노트를 검토하세요. 2.0 mol/L NaCl 원액으로 0.15 mol/L 용액 250 mL를 만들려 합니다. 동료는 원액 18.75 mL에 물 250 mL를 더하자고 했습니다. 필요한 원액량, 최종 부피를 맞추는 올바른 절차와 동료 절차의 실제 농도를 계산하세요. 부피 가산성을 가정한 근사 계산임을 표시하고 M·mL·L 단위를 혼동하지 마세요.''',
['Stock 18.75 mL.', 'Dilute to total 250 mL, not add 250 mL water.', 'Incorrect mixture .0375 mol/.26875 L ≈ .1395 mol/L under additive-volume assumption.'])

build = (ROOT/'build_vocab.py').read_text()
add('code_shortlist','coding',
'''아래는 실제 저장소의 draft shortlist 생성 코드입니다. 이를 재사용 가능한 순수 Python 함수 select_ids(valid, protected, freq, size)로 분리하려 합니다. 표준 라이브러리만 사용하세요. valid/protected는 정수 ID iterable, freq는 ID→빈도 mapping입니다. 입력은 수정하지 않습니다. valid 중 protected는 모두 포함하되 protected에 invalid ID가 하나라도 있으면 ValueError. size는 bool이 아닌 정수이며 len(protected) 이상 len(valid) 이하, 아니면 ValueError. 빈도는 양의 유한 실수만 사용하며 bool·NaN·inf·0·음수·invalid ID는 무시합니다. 나머지는 빈도 내림차순, 동률 ID 오름차순으로 선택하고 남는 자리는 valid ID 오름차순으로 채웁니다. 결과는 정렬된 list입니다. 원본의 연속 ID 가정을 제거하고 빈 valid/size=0도 처리하세요. 설명 대신 실행 가능한 함수와 필요한 import만 반환하세요.\n원본:\n'''+build,
['Executes all sparse/protected/size edge-case tests.', 'Executes deterministic frequency/tie and invalid-frequency tests.', 'Does not mutate inputs and returns exact sorted cardinality.'], thinking=True,limit=4096)
memory_source = (ROOT/'vocab_memory.py').read_text()
add('code_telemetry','coding',
'''다음 실제 측정 하네스와 호환되는 순수 Python 집계 함수를 새로 작성하세요: summarize_samples(samples, baseline_available). samples는 dict iterable이며 각 행의 phase(str), monotonic(초), MemAvailable(바이트)를 사용합니다. phase는 loading/benchmark/post_idle만 허용. 시간과 메모리는 bool 아닌 유한 숫자, 메모리는 0 이상이어야 하며 불량 행은 무시합니다. 중복 (phase, monotonic)은 입력에서 마지막 유효 행을 사용합니다. 각 phase별 count, min_available, median_available, max_extra_memory=max(0, baseline_available-min_available), max_gap(정렬된 인접 시간차 최대, 1행이면0)을 dict로 반환하고 빈 phase는 생략합니다. baseline은 bool 아닌 유한 비음수 숫자, 아니면 ValueError. samples와 내부 dict를 수정하지 마세요. 표준 라이브러리만 허용하며 함수와 import만 반환하세요.\n기존 하네스:\n'''+memory_source,
['Executes filtering, duplicate and invalid-baseline tests.', 'Correct phase medians, clamped deltas and sorted time gaps.', 'Input immutability and one/zero-row behavior correct.'], thinking=True,limit=4096)
normalize = (REPO/'util/talk/internal/orchestrator/assets/support-services/cmd/collector/normalize.go.asset').read_text()
add('code_html_review','coding',
'''실제 HTML 수집기의 다음 코드를 리뷰하세요. 요구사항은 script/style/template 등 숨겨진 하위 트리의 텍스트뿐 아니라 링크와 표도 결과에 들어가지 않는 것입니다. 현재 hidden 처리만으로 충족되는지 제어 흐름을 따라 판단하고, 재현 HTML과 예상 결과, 수정 위치와 회귀 테스트를 제시하세요. 공개 링크/표 처리와 nested-table 문제를 구분하고 무관한 리팩터링은 피하세요. 한국어로 설명하고 필요한 Go 수정 부분을 보여 주세요.\n'''+normalize,
['Finds links/tables collected despite hidden flag because switch executes before hidden filtering.', 'Proposes skipping hidden subtree before collection and preserves visible content.', 'Includes template-hidden link/table regression and positive visible controls.'],limit=2048)

report = (ROOT/'results/2026-09-08-vocab-memory/README.md').read_text()
summary = (ROOT/'results/2026-09-08-vocab-memory/summary.json').read_text()
add('research_memory','research',
'''첨부는 실제 같은 장비의 draft vocabulary 메모리 측정 보고서와 원시 집계입니다. 운영 검토 메모를 작성하세요. 128K가 64K보다 얼마나 더 메모리를 쓰는지, full이 왜 더 적을 수 있는지, 최고점과 상주량이 다른 이유, 1회 속도 차이의 해석, production 자동 KV에 그대로 적용 가능한지 분석하세요. 공유 weight를 이중 계산하거나 vocab 크기를 context 크기로 혼동하지 마세요. 근거 수치와 추가 검증 우선순위를 포함하세요.\n보고서:\n'''+report+'\n집계:\n'+summary,
['Correctly contrasts ~320 MiB exact head delta / ~0.4 GiB observed resident delta and full shared head.', 'Explains common loading peaks and distinction from steady-state/baseline-increment memory.', 'Notes fixed KV 65,536 and single-pass speed evidence cannot settle production performance.'], thinking=True,limit=4096)

add('human_translation','language',
'''다음 기술 고객 안내를 자연스러운 한국어로 번역하고, 이어서 비전문가용 세 문장 요약을 쓰세요. timeout, retry, idempotency key를 혼동하지 말고 가능성과 확정을 구분하세요. 원문: "A timeout does not necessarily mean that the operation failed. The server may have committed the transaction before the response was lost. Retrying a non-idempotent request without the original idempotency key can create a duplicate charge. Reuse the same key for retries of the same logical operation; use a new key for a genuinely new purchase. A successful refund request is not a guarantee that the funds will appear immediately. Settlement time depends on the payment provider."''',
['Timeout may occur after committed transaction, not definite failure.', 'Same logical operation same key; new purchase new key, duplicate-charge risk preserved.', 'Refund acceptance distinct from settlement arrival; three-sentence plain-language summary.'])
add('human_history','humanities',
'''사료 독해 과제입니다. 실제 역사 사건에 대한 자료가 아니라 가상 도시 기록입니다. [A: 1910년 시청 연감] 새 수도 시설을 통해 모든 구역에 물이 공급되었고 주민이 환영했다. [B: 1911년 외곽 주민 편지] 본관은 완공됐지만 우리 골목 연결은 아직 없고 물값 부담 때문에 신청하지 못했다. [C: 1930년 회고록] 당시 모두가 수도를 즉시 사용할 수 있었다고 기억한다. [D: 1910년 예산서] 본관 공사 완료, 가정 연결 보조금은 다음 회계연도로 이월. 자료 간 일치·충돌, 작성 시점·목적, '시설 완공'과 '실질 접근'의 차이를 논하고 확인할 추가 사료를 제시하세요. 자료 밖 인물·도시를 만들어 넣지 마세요.''',
['Distinguishes main completion from household connection/affordability.', 'Weighs contemporary letter/budget versus official framing and later recollection without blanket dismissal.', 'Proposes relevant corroborating records and no invented historical particulars.'])
add('human_writing','language',
'''다음 사실만으로 동료에게 보내는 한국어 일정 변경 메일을 작성하세요. 원래 금요일 15시 데모. 인증 모듈 통합 테스트에서 중복 결제 위험 발견. 수정은 끝났지만 회귀검증 미완료. 고객 데이터 유출 증거는 없음. 새 제안은 다음 주 월요일 11시, 참석 가능 여부는 목요일 17시까지 회신 요청. 책임 전가나 '보안 사고가 발생했다'는 과장을 피하고, 데모 지연 이유·현재 상태·요청 행동을 6문장 이내로 담으세요.''',
['Preserves duplicate-charge risk and fixed-but-unverified distinction.', 'Exact Monday 11 and Thursday 17 response deadline; <=6 body sentences.', 'No fabricated data breach or blame; clear collegial action request.'])

messages=[dict(role='user',content='가상의 가족 주말 계획을 함께 정리하자. 총예산 18만원, 어른 둘과 아이 하나. 자동차가 없고 실내 활동을 선호해. 외부 장소 검색은 하지 마.'),dict(role='assistant',content='제공하는 후보와 비용, 이동 조건을 기준으로 비교하겠습니다.'),dict(role='user',content='후보 A 실내 과학관은 입장 총 45,000원, 왕복 교통 18,000원, 점심 36,000원. 후보 B 야외 체험은 입장 60,000원, 교통 30,000원, 점심 42,000원. 두 후보 모두 선택적 기념품은 20,000원이고 예약 변경 수수료가 각각 5,000원이야.'),dict(role='assistant',content='기념품과 변경 여부를 정하면 총액을 계산할 수 있습니다.'),dict(role='user',content='예산을 12만원으로 줄였어. 비 예보라 야외는 제외. 기념품은 안 사고, 기존 예약 시간을 바꾸므로 수수료는 내야 해. 예전 18만원 조건을 사용하지 말고 최종 선택·항목별 합계·남는 예산을 정리해줘. 검색하지 말고 이 대화의 정보만 써.')]
add('dialogue_constraints','planning','', ['Selects indoor A and excludes outdoor B.', 'Total 104,000 including 5,000 change fee and no souvenir.', 'Uses latest 120,000 budget; remaining 16,000.'],messages=messages)

# Long heterogeneous records with genuine aggregation dependencies, not repeated filler.
records=[]
for i in range(180):
    region=['동부','서부','남부'][i%3]
    count=10+i%7
    records.append(dict(id=f'R{i:03}',region=region,units=count,unit_price=1200+(i%5)*100,status='cancelled' if i%11==0 else 'paid',memo=['정기 주문','주소 확인 완료','묶음 배송','재고 이월 없음'][i%4]))
expected={r:dict(orders=0,units=0,revenue=0) for r in ['동부','서부','남부']}
for row in records:
    if row['status']=='paid':
        e=expected[row['region']];e['orders']+=1;e['units']+=row['units'];e['revenue']+=row['units']*row['unit_price']
add('long_ledger','documents',
'''아래 가상 주문 원장을 감사하세요. cancelled는 모든 집계에서 제외하고 paid만 지역별 주문 수·수량 합계·매출(수량×단가)을 구하세요. 중복 기록은 없습니다. 결과를 JSON 객체로 먼저 쓰고 그 뒤 집계 방법과 검산 방법을 설명하세요. 합계나 중간 숫자를 어림잡지 마세요. JSON의 지역 키는 동부/서부/남부이며 각각 orders, units, revenue 정수 필드를 사용하세요.\n'''+json.dumps(records,ensure_ascii=False),
['Correct paid order counts per region.', 'Correct paid units per region.', 'Correct per-row-weighted revenue per region.'],limit=2048)
tasks[-1]['expected']=expected
assert len(tasks)==24, len(tasks)
assert sum(t['thinking'] for t in tasks)==4
for task in tasks:
    assert len(task['rubric'])==3
payload=dict(version=1,created='2026-09-08',kind='held-out controlled broad-domain scenarios',tasks=tasks)
raw=json.dumps(payload,ensure_ascii=False,indent=2)+'\n'
path=OUT/'tasks.json'
if path.exists():
    raise RuntimeError('Refusing to overwrite frozen tasks')
path.write_text(raw)
(OUT/'tasks.sha256').write_text(hashlib.sha256(raw.encode()).hexdigest()+'  tasks.json\n')
print(json.dumps(dict(tasks=len(tasks),domains=sorted(set(t['domain'] for t in tasks)),thinking=sum(t['thinking'] for t in tasks),max_output_tokens_per_pass=sum(t['max_tokens'] for t in tasks),ledger_expected=expected),ensure_ascii=False,indent=2))
