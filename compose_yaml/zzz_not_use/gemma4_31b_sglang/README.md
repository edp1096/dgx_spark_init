# Gemma 4 31B NVFP4 + EXL3 assistant (SGLang)

- 본체: `lyf/Huihui-gemma-4-31B-it-abliterated-v2-NVFP4`
- Assistant: `edp1096/gemma-4-31B-it-assistant-exl3` (body 4bit / head 6bit)
- GPU 추론, Frozen-KV MTP 7, BF16 KV, 256K context, 동시 생성 1개
- OpenAI API: `http://127.0.0.1:8000/v1`

본체·assistant의 고정 revision을 Hugging Face 캐시에 먼저 준비한다.
SparkTalk에서는 모델 자료 준비 기능을 사용한다. 실행 중에는 다운로드하지 않는다.

```bash
cp -n env.sample .env
docker compose up -d --build
docker logs -f sglang-gemma4-31b
```

Dockerfile은 고정된 SGLang 소스와 ExLlamaV3 1.5.4 GPU 확장을 빌드한다.
모델을 재변환하지 않는다. 기존 `sparktalk-qwen38fn_exl3:1.5.4-managed1` 이미지가
있으면 `docker build --target reuse -f Dockerfile.exl3 -t dgx-sglang-gemma4-31b-exl3mtp:2ef0fe4-exl3mtp1 .`로
이미 컴파일한 동일 확장을 재사용할 수 있다.

본체 NVFP4 실행은 SGLang이 담당하고, EXL3 GPU 커널은 assistant의 행렬에만 적용한다.
Gemma reasoning/tool parser, JSON schema, prefix cache를 유지한다.
이미지 tower의 양자화 제외 설정은 원본 파일을 수정하지 않고 런타임 config에 반영한다.
이미지 토큰은 프리필 chunk에서 나누지 않아 bidirectional attention을 유지한다.

Thinking은 SparkTalk의 켜짐/꺼짐과 요청별 thinking budget을 따른다.
전역 초기화 상한은 262144이며 요청별 예산이 이를 대체한다.

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/models
docker compose down
```
