# GLM 5.3 Flash NVFP4 · vLLM TP2

두 DGX Spark에서 NVIDIA 원본 또는 Huihui 변환본을 실행합니다. 기본은 **1M 문맥, FP8 KV 약 9.54GiB/호스트, DFlash2 K5, CUDA Graph와 비동기 스케줄링**입니다. 1M을 가득 채운 요청은 한 번에 하나만 수용합니다.

```bash
./manage.sh setup --official     # Huihui는 --abliterated
./manage.sh start
./manage.sh status
```

`env.sample`을 기준으로 `.env`의 두 호스트·RoCE 주소를 맞춥니다. 이미지는 고정된 Pilcothink 0.28 ARM64 버전을 자동으로 받고 워커에 전달합니다. 본체와 `incoai/GLM-5.3-Flash-DFlash2`를 함께 준비하며, 기존 완료된 캐시는 재사용합니다. 선택은 `MODEL_VARIANT=official|abliterated`, DFlash 비활성화는 `DFLASH_TOKENS=0`입니다. MTP는 별도이며 미검증이라 끕니다.

Huihui 실측: 일반 문장 32~36t/s, 코드 43~49t/s, 구조화 출력 47~50t/s(128 출력 토큰). 기능 9개·RGB 3종, 실제 1,047,622토큰 입력의 코드 5개 회수 통과. 1M 처리에는 약 19분 11초가 걸렸습니다. CPU 70%·GPU 2100MHz 제한에서 측정했습니다. 시작 과정의 스왑 쓰기는 남아 있습니다.

[검증 조건과 결과](docs/pilco-reproduction/README.md) · [Huihui 결과](docs/pilco-reproduction/huihui/spec-measured.md) · [Pilcothink 원본 레시피](https://github.com/gpdev-Pilcothink/DGX_Spark_vllm_Dockerfile/tree/main/0.28/GLM53-flash)
