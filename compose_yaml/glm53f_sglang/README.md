# GLM 5.3 Flash NVFP4 · SGLang TP2

두 DGX Spark에서 NVIDIA 원본 또는 Huihui abliterated 모델을 실행합니다. 기본값은 1M 문맥, 동시 요청 1개, FP8 KV, DFlash2 5토큰입니다. MTP와 디스크 KV 저장은 사용하지 않습니다.

```sh
cp env.sample .env
# .env의 모델 종류, 호스트와 RoCE 주소 확인
./manage.sh setup
./manage.sh start
```

로컬 이미지 `dgx-sglang-glm53:sm121-dev6`가 없으면 Dockerfile로 빌드하고 워커에 같은 이미지를 전송합니다. 모델은 한 번 다운로드한 뒤 워커로 복사합니다. 기존 모델을 중지하고 실행하세요.

`MODEL_VARIANT=official|abliterated`로 모델을 선택합니다. 문맥은 `MAX_MODEL_LEN`, DFlash는 `DFLASH_TOKENS=5|0`으로 설정합니다. Talk 내장 레시피도 같은 소스를 사용합니다. 현재 GLM 세트는 Magpie TTS를 제외하고 Extra 3종을 사용합니다. 아래 TTS 동시 실행 수치는 제외 전 검증 기록입니다.

Huihui + 워커 TTS·Extra 3종을 켠 상태에서 실제 입력 1,047,622토큰의 코드 5개를 정확히 회수했습니다(753.2초). 동일 입력의 Pilcothink vLLM+DFlash2는 1,151.3초였습니다. 각각 조정한 실행 설정의 비교이며, 엔진 자체의 보편적 우열을 뜻하지 않습니다.

128토큰 출력 벤치마크의 평균 생성 속도는 일반 문장 30.5, 코드 42.8, 구조화 출력 50.2 t/s였습니다. 기능·이미지 12개, 대화·캐시 26개, Talk 대화 API를 통과했습니다. CPU 70%·GPU 2100MHz 제한, LLM 동시 요청 1개 조건입니다.

스왑이 완전히 없지는 않습니다. 기동·워밍업 중 헤드/워커 약 5.29/3.51GiB, 1M 시험 중 0/519MiB가 기록됐습니다. 1M 처리 중 TTS도 실제 호출했습니다. OOM·재부팅 없이 완료했으며, 상세 조건과 원시 결과는 [qualification.json](qualification.json), [최종 검증](docs/production-qualification/)에 있습니다.

출처: [SGLang](https://github.com/sgl-project/sglang/tree/425a1f8f247d0cc17f2a3f3c2dba6c0bdf936552), [B12X](https://github.com/lukealonso/b12x), [Transformers](https://github.com/huggingface/transformers). 기반 이미지: [QAD Dockerfile](../qwen38_fn_sglang/Dockerfile).
