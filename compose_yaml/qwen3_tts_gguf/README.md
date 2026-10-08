# Qwen3-TTS Q8 for SparkTalk

Qwen3-TTS 0.6B CustomVoice의 본체와 오디오 codec을 모두 배포 Q8_0 GGUF로
실행한다. ServeurpersoCom/qwentts.cpp의 네이티브 `tts-server`를 CUDA13/GB10
용으로 빌드하며 `GGML_BACKEND=CUDA0`으로 GPU를 지정한다. CUDA가 없으면
초기화를 실패시키며 CPU 전용 서빙으로 전환하지 않는다.

- 서버 소스: `51512f129a7419567f4b8abfb06801451789b8f1`
- GGUF revision: `b7ee2e8c7459c3bea99da23e3d178125a7d1713c`
- 본체: `qwen-talker-0.6b-customvoice-Q8_0.gguf`
- 오디오 codec: `qwen-tokenizer-12hz-Q8_0.gguf`
- API 모델 이름: `qwen3-tts-0.6b-q8`, 기본 화자 `sohee`, 24kHz mono s16le
- 서버는 모델을 계속 로드한다. 기본 동시 합성은1개이며 HTTP 대기까지
  최대4개를 받아 상태 조회·종료 요청용 worker를 남긴다.

```bash
python3 scripts/setup_models.py
docker compose build api
docker compose up -d api
curl http://127.0.0.1:8692/ready
curl http://127.0.0.1:8692/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-tts-0.6b-q8","input":"영상은 팔육사 사팔공 초당 이십사 에프피에스 입니다.","language":"Korean","voice":"sohee","response_format":"pcm","seed":42}' \
  -o speech.pcm
```

`QWEN_TTS_MODEL_DIR`, `QWEN_TTS_PORT`, `QWEN_TTS_BIND_ADDR`로 경로와 주소를
지정한다. Talk의 모델 준비는 같은 revision/SHA256의 두 파일을 직접 받고
GGUF 변환 없이 사용한다. 모델 가중치는 Git·Docker 이미지에 포함하지 않는다.

PCM 응답은 생성 중 전송한다. `response_format: wav`는 전체 합성이 끝난 뒤
반환하는 별도 경로이므로 PCM과 같은 출력/지연이라고 취급하지 않는다.

수명 패치는 HTTP 입장과 `/v1/runtime/memory`, `/v1/runtime/quiesce`,
`/v1/runtime/resume`만 추가한다. 본체·codec 가중치와 추론 커널은 수정하지
않는다. 실행·대기·PCM 응답 소비 중에는 종료를 거절하고 유휴 상태에서만
quiesce를 허용한다. `/ready`는 모델 초기화가 끝난 후부터 응답한다.

Talk는 한국어 문장 속 브랜드명과 약어를 같은 문장으로 전달하고 로케일을
Qwen 언어 이름으로 변환한다. 긴 답변은 문장·단어 경계를 우선해 최대384자로
나누어 전달한다. 지원 언어는 한국어·영어·일본어·중국어·러시아어·
독일어·프랑스어·스페인어·이탈리아어·포르투갈어다. Q8도 고유명사 오류가
남으며, 해상도/fps의 원하는 읽기는 별도 정규화 정책이 필요하다.

출처: https://github.com/ServeurpersoCom/qwentts.cpp,
https://huggingface.co/Serveurperso/Qwen3-TTS-GGUF.
