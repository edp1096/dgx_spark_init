# ReActor face swap service

Spark Media용 독립 얼굴 교체 서비스입니다. `Gourieff/ComfyUI-ReActor`
0.7.0-alpha2의 `ReActorFaceSwap` 노드를 고정해 사용하며, DGX Spark의
aarch64/CUDA 13 환경에서는 ONNX Runtime GPU 1.29.0을 사용합니다.

```bash
cd compose_yaml/reactor_faceswap
docker compose up -d --build
curl http://127.0.0.1:8706/health
```

첫 시작에는 `inswapper_128`, `buffalo_l`, upstream SFW 검사 모델을
`media-reactor-models` 볼륨에 준비합니다. InsightFace 사전학습 가중치는
비상업 연구용 조건이므로 상업 서비스에는 별도 라이선스 모델이 필요합니다.

출처: https://github.com/Gourieff/ComfyUI-ReActor
