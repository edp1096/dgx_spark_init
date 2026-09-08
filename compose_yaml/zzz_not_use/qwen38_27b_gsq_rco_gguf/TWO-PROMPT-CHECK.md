# Qwen 27B EXL3·GGUF 반복 및 환각 검사 — 2026-09-08

현재 설치된 **Qwen3.8 27B 파생 모델** 두 개로 검사했다. 요청에 적힌 Qwen3.7은 설치 기록과 달라, 검사 시작 때 Qwen3.8 기준으로 진행한다고 안내했다.

## 결론

**EXL3에서도 심한 사실 환각이 확인됐다. 온도 1.0은 두 모델의 환각을 해결하지 못했다.**

- 자기소개 질문은 모든 조건에서 최종 답변을 냈으며, 긴 반복은 관찰되지 않았다.
- 궁궐 설명은 최종 답변이 나온 7개 Thinking 조건 모두 주요 역사 사실·건물·장소에 다수의 오류가 있었다.
- GGUF 0.7/xhigh는 생각 과정에서 `인정전, 대조전, 창덕전`이라는 묶음을 20회 반복한 뒤 **최종 답변이 빈 상태로 종료**했다.
- 나머지 Thinking 조건에서는 이런 긴 연속 반복이 관찰되지 않았다. EXL3도 환각은 심하지만, 이번 입력에서는 GGUF와 동일한 긴 반복·빈 답변 현상까지 재현되지는 않았다.
- 모든 요청의 API 종료 사유는 `stop`이었다. GGUF 반복 사례도 실제로 끝없이 실행된 것은 아니며, 8,192토큰 한도에 도달한 것도 아니다(생성 519토큰).

## 조건

각 모델·온도·Thinking 조합에서 별도 대화로 다음 두 질문을 순서대로 보냈다.

1. `너는 누구냐?`
2. `한국의 궁궐에 대해 설명.`

두 번째 요청은 첫 질문과 해당 모델의 첫 최종 답변을 포함했다. 별도 시스템 프롬프트, 검색, RAG, 도구는 넣지 않았다. SparkTalk UI가 아닌 엔진 API 직접 검사다.

- 온도: **0.7 / 1.0**
- Thinking: **medium / xhigh**
- 공통: `top_p=0.95`, `top_k=20`, `seed=42`, 문맥 131,072토큰, 최대 생성 8,192토큰
- GGUF: Huihui 기반 ISTA IQ3_S Allocation MTP, llama.cpp e71b805, MTP 3, Q8 KV, 프롬프트 캐시 1GiB
- EXL3: Lygodactylus/Qwen3.8-27B-Uncensored-exl3-4bpw, ExLlamaV3 63b32f0, MTP, NVFP4 KV
- 기존 EXL3 API가 `reasoning_effort`를 전달하지 않아 **검사용 복사본에서 해당 인자만 템플릿으로 전달**하도록 연결했다. 서버 로그에서 medium에는 xhigh 지시문이 없고, xhigh에는 있는 것을 확인했다. 보관된 EXL3 서비스 파일은 수정하지 않았다.
- 최초에는 Thinking 꺼짐·온도 0.7·최대 2,048토큰으로 두 모델을 검사했다. 이후 사용자의 medium/xhigh 및 온도 1.0 요청을 반영해 위 16개 요청을 추가했다. 총 20개 응답을 보존했다.

## 궁궐 설명 결과

| 모델 | 온도 | Thinking | 반복·종료 | 사실 정확도 | 전체 시간 |
|---|---:|---|---|---|---:|
| GGUF | 0.7 | medium | 긴 연속 반복 없음; 5대 궁궐 표에 창덕궁 중복 | 심함: 경운궁=창덕궁, 창경궁이 경희대 교내라는 설명 | 40.51초 |
| GGUF | 0.7 | xhigh | 생각 과정에서 같은 전각 묶음 20회 반복; 최종 답변 없음 | 최종 답변 평가 불가; 생각 과정에도 전각 혼동 | 20.59초 |
| EXL3 | 0.7 | medium | 긴 연속 반복 미관찰 | 심함: 덕수궁 대신 건청궁, 건청궁을 1805년 순조가 아버지(정순왕후)를 위해 건립했다고 설명 | 50.86초 |
| EXL3 | 0.7 | xhigh | 긴 연속 반복 미관찰 | 심함: 창경궁 1476년 세조, 경희궁 1405년 태종, 덕수궁 개칭 1900년 | 63.39초 |
| GGUF | 1.0 | medium | 긴 연속 반복 미관찰 | 심함: 창덕궁 1405년 경종, 창경궁 1448년 중종, 경희궁이 경희대 캠퍼스라는 설명 | 48.13초 |
| GGUF | 1.0 | xhigh | 긴 연속 반복 미관찰 | 심함: 5대 궁궐에 창원궁, 궁궐들이 2009년 세계유산으로 묶여 등재됐다는 설명 | 72.78초 |
| EXL3 | 1.0 | medium | 긴 연속 반복 미관찰 | 심함: 경희궁을 1905년 덕수궁으로 개칭, 창경궁 1405년, 창덕궁 등재 2000년 | 56.30초 |
| EXL3 | 1.0 | xhigh | 긴 연속 반복 미관찰 | 심함: 창덕궁이 임진왜란에 소실되지 않았다는 설명, 1834년 고종 즉위, 경희궁이 경희대 캠퍼스라는 설명 | 39.89초 |

GGUF 온도 1.0에서는 `정무를处理的한`, `즉后来的` 같은 부자연스러운 언어 혼입도 나타났다. 반복 유무와 사실 정확도는 별도로 평가했다. 정상 `stop`과 읽기 좋은 표·목록은 내용의 정확성을 보장하지 않았다.

## 대표 오류 대조

- **5대 궁궐 목록:** 경복궁·창덕궁·창경궁·경희궁·경운궁(덕수궁)이 기준이다. 양화궁·창원궁을 끼워 넣거나 창덕궁을 중복 계산한 답변은 잘못이다. [서울시 역사 안내](https://www.seoul.go.kr/seoul/history.do)
- **창덕궁:** 1405년 태종 때 창건, 1592년 소실, 1610년 중건이다. 경종·정종이 1405년에 세웠다거나 임진왜란에 타지 않았다는 설명은 틀렸다. [궁능유적본부 창덕궁 소개](https://royal.khs.go.kr/ROYAL/contents/R102010000.do?menuId=03_02_01)
- **창경궁:** 성종 때인 1483년 수강궁을 확장했으며 1484년 완공됐다. 1405년·1447년·1448년·1476년·1610년 창건이라는 답변은 모두 오류다. [국가유산청 창경궁 설명](https://digital.khs.go.kr/heri/heriDetail.do?ctptNo=1331101230000&ctptUid=13898859674569300744)
- **덕수궁:** 경운궁에서 덕수궁으로 이름을 바꾼 해는 1907년이다. 경희궁이 1905년에 덕수궁으로 바뀌었다는 설명은 궁궐과 연도 모두 잘못됐다. [궁능유적본부 덕수궁 역사](https://royal.khs.go.kr/ENG/contents/E104010000.do)
- **건청궁:** 경복궁 내에 1873년 조성됐다. 1805년에 순조가 지었다는 설명과 5대 궁궐에 독립적으로 포함한 분류는 잘못됐다. [궁능유적본부 경복궁 역사](https://royal.khs.go.kr/ROYAL/contents/R101010000.do?menuId=03_01_01)
- **경희궁:** 광해군 때 조성된 궁궐이며 서울역사박물관 분관으로 소개된다. 경희대학교 캠퍼스라는 설명은 잘못됐다. [서울역사박물관 경희궁 역사](https://museum.seoul.go.kr/www/intro/annexIntro/annex_20/annex_20_03.jsp?sso=ok)
- **세계유산:** 창덕궁은 1997년 등재됐다. 경복궁·창덕궁·창경궁·창원궁이 2009년에 하나의 세계유산으로 등재됐다는 설명은 사실이 아니다. [UNESCO 창덕궁](https://whc.unesco.org/en/list/816/)

## 답변 원문

각 링크에는 최종 답변과 펼쳐 볼 수 있는 생각 과정 원문이 있다. 오류를 교정하거나 문장을 다듬지 않았다. 같은 폴더의 request.json과 SSE에는 실제 요청과 스트림도 보존했다.

| 모델 | 온도 | Thinking | 첫 질문 | 둘째 질문 |
|---|---:|---|---|---|
| GGUF | 0.7 | medium | [자기소개](data/two-prompts-20260908/thinking/gguf/medium/1-transcript.md) | [궁궐 설명](data/two-prompts-20260908/thinking/gguf/medium/2-transcript.md) |
| GGUF | 0.7 | xhigh | [자기소개](data/two-prompts-20260908/thinking/gguf/xhigh/1-transcript.md) | [궁궐 설명](data/two-prompts-20260908/thinking/gguf/xhigh/2-transcript.md) |
| EXL3 | 0.7 | medium | [자기소개](data/two-prompts-20260908/thinking/exl3/medium/1-transcript.md) | [궁궐 설명](data/two-prompts-20260908/thinking/exl3/medium/2-transcript.md) |
| EXL3 | 0.7 | xhigh | [자기소개](data/two-prompts-20260908/thinking/exl3/xhigh/1-transcript.md) | [궁궐 설명](data/two-prompts-20260908/thinking/exl3/xhigh/2-transcript.md) |
| GGUF | 1.0 | medium | [자기소개](data/two-prompts-20260908/temperature1/gguf/medium/1-transcript.md) | [궁궐 설명](data/two-prompts-20260908/temperature1/gguf/medium/2-transcript.md) |
| GGUF | 1.0 | xhigh | [자기소개](data/two-prompts-20260908/temperature1/gguf/xhigh/1-transcript.md) | [궁궐 설명](data/two-prompts-20260908/temperature1/gguf/xhigh/2-transcript.md) |
| EXL3 | 1.0 | medium | [자기소개](data/two-prompts-20260908/temperature1/exl3/medium/1-transcript.md) | [궁궐 설명](data/two-prompts-20260908/temperature1/exl3/medium/2-transcript.md) |
| EXL3 | 1.0 | xhigh | [자기소개](data/two-prompts-20260908/temperature1/exl3/xhigh/1-transcript.md) | [궁궐 설명](data/two-prompts-20260908/temperature1/exl3/xhigh/2-transcript.md) |

## 최초 Thinking 꺼짐 검사

GGUF는 ‘양화궁’을 5대 궁궐에 넣는 등 심한 환각을 보였다. EXL3도 창덕궁·창경궁·경희궁·덕수궁에 `1405년 태종이 세운 동궁`이라는 잘못된 설명을 반복해서 붙였다. 이는 여러 항목에서 잘못된 문구를 재사용한 것으로, 생각 과정에서 동일 묶음을 연속 출력한 GGUF xhigh 사례와 구분했다.

- [GGUF 자기소개](data/two-prompts-20260908/gguf/1-transcript.md) · [GGUF 궁궐 설명](data/two-prompts-20260908/gguf/2-transcript.md)
- [EXL3 자기소개](data/two-prompts-20260908/exl3/1-transcript.md) · [EXL3 궁궐 설명](data/two-prompts-20260908/exl3/2-transcript.md)

## 해석 범위와 재현 자료

각 조건은 같은 시드로 한 번 실행했다. 반복 발생률이나 일반적인 오류율을 추정한 검사가 아니며, ‘미관찰’은 재발하지 않는다는 뜻이 아니다. 두 가중치는 서로 다른 언센서드 파생 모델이고 양자화·엔진도 다르므로, 결과 차이를 GGUF/EXL3 파일 형식 하나의 원인으로 분리할 수 없다. 온도 1.0에서 반복이 사라진 한 사례만으로 해결됐다고 판단할 수도 없다.

실제 운영 설정을 바꾸지 않았고, 임시 서버는 모두 제거하여 검사 시작 당시의 GPU 서버 정지 상태로 복원했다.

[전체 측정 요약 JSON](data/two-prompts-20260908/summary.json) · [0.7 검사 스크립트](data/two-prompts-20260908/run-thinking.py) · [1.0 검사 스크립트](data/two-prompts-20260908/run-temp1.py) · [EXL3 검사용 인자 연결](data/two-prompts-20260908/exl3-thinking.py)

## NVFP4 추가 검사

[NVFP4 동일 조건 추가 검사](NVFP4-TWO-PROMPT-CHECK.md)에서도 네 조건 모두 심한 사실 오류가 확인됐다. 긴 연속 반복·빈 답변은 해당 검사에서 관찰되지 않았다.
