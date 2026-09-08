# Qwen3.8 27B · Huihui BF16 → GSQ-RCO GGUF 사전 점검

목표는 보유한 Huihui abliterated BF16에서 후보 텐서를 새로 최적화하고 RCO를
실행하여 IQ3_S 수준의 용량을 가진 GGUF를 만드는 것이다. MTP 및 비전도 검증 대상이다.

**초기 조사 결과: BF16 입력 검증 완료, 전체 GSQ 최적화·RCO 탐색 재실행 경로는 미확보.**
이 문서는 그 사전 조사 기록이다. 이후 사용자가 승인한 배분표 재사용 방식의
실행 레시피와 생성 결과는 [README.md](README.md), [EVALUATION.md](EVALUATION.md)에 있다.

## 확인한 입력

| 모델 | 고정 리비전 | 가중치 |
|---|---|---|
| Qwen/Qwen3.8-27B | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | BF16, 18 shards, 1,199 tensors, 약 55.56 GB |
| huihui-ai/Huihui-Qwen3.8-27B-abliterated | `d42ca8978c5a66e92c3446d46e8adfe03ef692ff` | BF16, 18 shards, 1,199 tensors, 약 55.56 GB |

두 입력 모두 로컬에 있으며 MTP·비전 텐서와 토크나이저가 있다. 인덱스와 실제 헤더,
텐서 크기·파일 길이를 검사했고 두 모델의 텐서 이름·shape·dtype 스키마가 일치했다.
전체 가중치 payload 해시를 검사하거나 GPU에 모델을 로드한 검증은 아니다.

```sh
python3 preflight.py \
  --official "$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0" \
  --abliterated "$HOME/.cache/huggingface/hub/models--huihui-ai--Huihui-Qwen3.8-27B-abliterated/snapshots/d42ca8978c5a66e92c3446d46e8adfe03ef692ff" \
  --output input-manifest.local.json
```

## 2026-09-08 공개 도구 조사

- GSQ main: `03fc16484c369e3127225615d5e03e8d3a6043e3`.
  공개 경로는 HF/packed checkpoint 및 Humming 계열이다.
- RCO main: `9a1e09c07d468109cbe60a1b87d5036034a79d10`.
  공개 양자화 예제는 GPTQ 후보 DB → 비트폭 탐색 → HF checkpoint이다.
- GSQ의 `gsvq-iquant-qwen3-sweep-results` 브랜치:
  `c93f2e81a1cf8070ee7c32bec7f787d0ed7f496f`.
  기존 IQuant 파일의 고정 scale/코드 수정 실험이 있고, 문서상 재구성 오차 개선이
  전체 모델 KL 개선으로 이어지지 않은 사례도 명시되어 있다. 출시 27B 모델의
  GSQ-RCO 제작 경로와 동일하다는 근거는 없다.
- HF release: `ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF`,
  확인 시 리비전 `d562806dbafae37109975e970aae91b43e73b440`.
  완성 GGUF·imatrix·최종 allocation 자료는 있으나 원본 변경 시 필요한 후보 생성 및
  탐색/조립 전체 구현을 제공하지 않는다.

진행에 필요한 미확보 구성요소:

1. 출시 모델에서 사용하는 native GGUF qtype별 GSQ 후보 생성·최적화 경로.
2. 해당 후보들의 실제 직렬화 크기를 반영하는 비용 모델과 RCO 연결.
3. Qwen3.8-27B 하이브리드 구조·embedding·출력층·MTP 처리 설정 및 calibration recipe.
4. 선택된 후보를 다시 일반 양자화하지 않고 GGUF에 조립하는 검증된 export 경로.

제작자의 [GGUF 변환 관련 답변](https://github.com/IST-DASLab/GSQ/issues/4#issuecomment-4830789814)은
당시 공개 코드로 해당 GGUF를 만들 수 없고 별도 코드 공개가 필요하다고 설명한다.
최근의 [27B 출시 모델 재현 도구 요청](https://github.com/IST-DASLab/GSQ/issues/9)도
같은 구성요소를 요청하며, 확인 시 답변은 없었다. 본 작업에서 이슈나 메시지를
새로 전송하지 않았다.

## 재개 조건

해당 구현/recipe가 공개되거나 확보되면 위 BF16 원본으로 일부 층 검증을 먼저 한다.
형식·오차·피크 메모리·실행 시간을 확인한 뒤 전체 후보 생성과 탐색을 수행한다.
그 전에는 일반 GGUF 변환, 기존 allocation 재사용, 실험용 GSVQ 결과를
GSQ-RCO 재현 결과로 취급하지 않는다. 필요한 GPU 자원과 소요 시간은 아직 미확정이다.

최종 검증은 공식 GSQ-RCO 및 기존 NVFP4·EXL3와 한국어/코드/도구 호출·비전 품질을
비교하고, MTP off/on의 실제 채택률·속도·메모리를 측정한다.

## 별도 실험 대안

사용자가 제시한 RentedNoodle 모델의 [제작 방식 확인](COMMUNITY.md)을 추가했다.
ISTA의 최종 배분표를 Huihui에 재사용하는 접근이며, 위 전체 GSQ-RCO 재현과 구분한다.
