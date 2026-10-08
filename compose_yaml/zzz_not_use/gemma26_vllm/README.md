> 이전 vLLM 구성 보관본. 현재 Gemma 운영 구성은 [gemma26_sglang](../../gemma26_sglang/compose.yaml)이다.

# Huihui Gemma 4 26B · TP1

단일 DGX Spark용 NVFP4. 1M 문맥은 proportional RoPE 위치 보간 실험이며,
**현재 변환·실측 진행 중이다.** 원래 문맥은 262,144 토큰이다.

```bash
./manage.sh start
./manage.sh logs
./manage.sh stop
```

기본 모델 경로: `~/.cache/huggingface/edp1096/Huihui-Gemma-4-26B-A4B-it-NVFP4`.
공통 이미지 빌드는 `./manage.sh setup`. [구현·작업 상태](runtime/docs/WORK_STATUS.md).

출처: [Google 원본](https://huggingface.co/google/gemma-4-26B-A4B-it),
[BG NVFP4](https://huggingface.co/bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4),
[Huihui BF16](https://huggingface.co/huihui-ai/Huihui-gemma-4-26B-A4B-it-abliterated).
