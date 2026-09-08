# Qwen3.8 27B NVFP4 반복·환각 추가 검사 — 2026-09-08

**NVFP4에서도 네 조건 모두 궁궐 설명에 심한 사실 환각이 있었다.** 이번 검사에서는 긴 연속 반복이나 빈 최종 답변은 관찰되지 않았다. 온도 1.0에서도 오류가 이어졌다.

## 검사 조건

[앞선 EXL3·GGUF 검사](TWO-PROMPT-CHECK.md)와 같은 두 질문을 각 조건의 새 대화에서 순서대로 입력했다.

1. `너는 누구냐?`
2. `한국의 궁궐에 대해 설명.`

두 번째 요청에는 첫 질문과 해당 모델의 첫 최종 답변을 포함했다. 시스템 프롬프트·검색·RAG·도구 없이 엔진 API를 직접 호출했다.

- 온도 **0.7 / 1.0** × Thinking **medium / xhigh**, 총 4조건·8응답
- `top_p=0.95`, `top_k=20`, `seed=42`, 최대 생성 8,192토큰
- 문맥 131,072토큰: 서버 모델 정보에서도 확인
- 모델: `Huihui-RadixArk-Qwen3.8-27B-abliterated-NVFP4` (혼합 양자화)
- 엔진: SGLang `2ef0fe4`, DFlash2 8토큰, FP8 KV, 이전 사용 프로필 기반
- `chat_template_kwargs`의 Thinking 단계가 SGLang의 요청 처리에서 템플릿으로 전달되는 코드를 확인했고, 모델 템플릿의 medium/xhigh 렌더링 차이도 확인했다. [렌더링 원문](data/two-prompts-20260908/nvfp4/templates.json)

## 결과

자기소개 네 응답은 모두 Qwen으로 자신을 소개하며 종료했고, 반복이나 빈 답변은 없었다. 궁궐 설명 결과는 다음과 같다.

| 온도 | Thinking | 긴 연속 반복 | 최종 답변 | 대표적인 환각 | 전체 시간 |
|---:|---|---|---|---|---:|
| 0.7 | medium | 미관찰 | 있음, `stop` | 창경궁 1405년·서궁, 경희궁 1421년, 덕수궁 개칭 1927년 | 71.27초 |
| 0.7 | xhigh | 미관찰 | 있음, `stop` | 창덕궁 1405년 세조, 창경궁 성북구·국립중앙박물관 부지, 경희궁과 덕수궁이 원래 같은 공간이라는 설명 | 62.31초 |
| 1.0 | medium | 미관찰 | 있음, `stop` | 5대 궁궐에 규회궁 포함, 창경궁 1470년·임진왜란에 유일하게 불타지 않았다는 설명 | 75.08초 |
| 1.0 | xhigh | 미관찰 | 있음, `stop` | 창경궁 1609년 창건, 경희궁 16세기 중반 창건, 경복궁·창덕궁 세계유산 등재 2001년 | 68.11초 |

모든 응답은 `stop`으로 종료됐으며, 출력 한도까지 계속 반복하거나 생각 과정만 남기고 최종 답변을 내지 못한 사례는 없었다. 표·목록과 문장이 만들어지는 상태 자체는 정상이지만, 궁궐 이름·창건 시기·위치·전각·문화유산 정보를 잘못 조합했다.

## 대표 오류 대조

- 덕수궁으로 이름을 바꾼 해는 **1927년이 아니라 1907년**이며, 이전 이름은 경운궁이다. 경희궁과 같은 공간이었던 것도 아니다. [궁능유적본부 덕수궁 역사](https://royal.khs.go.kr/ENG/contents/E104010000.do)
- 창경궁은 **1483년 성종 때** 수강궁을 확장해 조성했고, 1592년 임진왜란 때 소실됐다. 주소도 **서울 종로구**다. [궁능유적본부 창경궁 소개](https://royal.khs.go.kr/ROYAL/contents/R103010000.do)
- 창덕궁은 **1405년 태종 때** 창건됐다. 세조가 창건했다는 설명은 틀렸다. [궁능유적본부 창덕궁 소개](https://royal.khs.go.kr/ROYAL/contents/R102010000.do?menuId=03_02_01)
- 경희궁은 광해군 때 조성된 궁궐이다. 1411년·1421년·16세기 중반 창건이라는 답변은 잘못이다. [서울역사박물관 경희궁 역사](https://museum.seoul.go.kr/www/intro/annexIntro/annex_20/annex_20_03.jsp?sso=ok)
- 5대 궁궐은 **경복궁·창덕궁·창경궁·경희궁·경운궁(덕수궁)**이다. ‘규회궁’을 넣은 목록은 틀렸다. [서울시 역사 안내](https://www.seoul.go.kr/seoul/history.do)
- 창덕궁의 세계유산 등재는 **1997년**이다. 경복궁과 창덕궁이 2001년에 함께 등재됐다는 설명은 사실이 아니다. [UNESCO 창덕궁](https://whc.unesco.org/en/list/816/)

## 세 구성 비교

| 구성 | 이번 검사에서 긴 연속 반복·빈 답변 | 궁궐 설명의 사실 정확도 |
|---|---|---|
| GGUF | 0.7/xhigh에서 전각 묶음 20회 반복 후 최종 답변 없음 | 답변이 나온 나머지 조건도 다수의 심한 오류 |
| EXL3 | 미관찰 | 네 Thinking 조건 모두 심한 오류 |
| NVFP4 | 미관찰 | 네 Thinking 조건 모두 심한 오류 |

이 입력에서는 **세 구성 모두 환각 문제를 보였고, NVFP4로 바꾸는 것만으로 정확도가 해결된다는 근거는 나오지 않았다.** 다만 각 조건 한 번·고정 시드 검사이므로 반복 발생률을 추정하거나 ‘절대 반복하지 않는다’고 결론 내릴 수는 없다. 가중치 파생본·양자화·엔진·가속 방식이 달라 파일 형식이나 특정 가속 기능 하나의 원인으로 분리한 실험도 아니다.

## 전체 답변 원문

최종 답변과 펼쳐 볼 수 있는 생각 과정 원문을 그대로 저장했다. 동일 폴더의 request.json 및 SSE 파일에서 실제 요청·스트림을 확인할 수 있다.

| 온도 | Thinking | 자기소개 | 궁궐 설명 |
|---:|---|---|---|
| 0.7 | medium | [첫 답변](data/two-prompts-20260908/nvfp4/0.7/medium/1-transcript.md) | [둘째 답변](data/two-prompts-20260908/nvfp4/0.7/medium/2-transcript.md) |
| 0.7 | xhigh | [첫 답변](data/two-prompts-20260908/nvfp4/0.7/xhigh/1-transcript.md) | [둘째 답변](data/two-prompts-20260908/nvfp4/0.7/xhigh/2-transcript.md) |
| 1.0 | medium | [첫 답변](data/two-prompts-20260908/nvfp4/1.0/medium/1-transcript.md) | [둘째 답변](data/two-prompts-20260908/nvfp4/1.0/medium/2-transcript.md) |
| 1.0 | xhigh | [첫 답변](data/two-prompts-20260908/nvfp4/1.0/xhigh/1-transcript.md) | [둘째 답변](data/two-prompts-20260908/nvfp4/1.0/xhigh/2-transcript.md) |

[측정 요약 JSON](data/two-prompts-20260908/nvfp4/summary.json) · [실행 스크립트](data/two-prompts-20260908/nvfp4/run.py) · [실행 인자](data/two-prompts-20260908/nvfp4/launch.json) · [서버 로그](data/two-prompts-20260908/nvfp4/server.log)

검사용 서버는 종료·제거했고, 검사 시작 당시의 GPU 서버 정지 상태로 복원했다. 운영 모델·온도·Thinking 설정은 변경하지 않았다.
