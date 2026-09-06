# TP1 비교 도구

이미 실행 중인 로컬 Flash-Next API를 비교한다. 컨테이너를 시작하거나 종료하지
않으며, 사용자 대화나 SparkTalk 데이터베이스를 읽지 않는다.

```sh
python3 compare.py --output /tmp/flash-next-baseline.json
python3 compare.py --corpus --output /tmp/flash-next-corpus.json
```

기본 시험은 한국어 설명·이야기, Python·Go 코드, JSON, 영어 설명을 각각 두 번
실행한다. 요청마다 prefix cache를 비워 조건을 맞춘다. `decode_tok_s`는 스트림
첫 내용부터 마지막 내용까지의 시간과 서버가 반환한 completion token 수로
계산한다. MTP 스트림은 한 번에 여러 토큰을 전송하므로 이는 개별 토큰의 지연
측정과 다르다. TTFT, 출력, 종료 사유, 토큰 수, 최소 시스템 가용 메모리도 저장한다.

추가로 thinking ON/OFF 도구 호출, 빨강·파랑 이미지, 동시 요청 2개, 긴 문맥에서의
정답 검색을 검사한다. 모든 프롬프트와 이미지는 합성 데이터다. 정답 검색 한 건은
장문 품질 평가 전체를 대신하지 않는다.

`--corpus`는 속도 시험과 별개의 프롬프트로 초안 어휘 선정 자료를 생성한다.
서버 이미지의 Python 환경에서 어휘 목록을 만들 수 있다.

```sh
python3 build_vocab.py \
  --tokenizer /models/target/tokenizer.json \
  --corpus /tmp/flash-next-corpus.json \
  --verify /tmp/flash-next-baseline.json \
  --output /tmp/draft-vocab.json
```

`draft-vocab.json`에는 토크나이저·자료의 SHA256, 선정 규칙, 토큰 ID 범위와
포함률이, 같은 이름의 `.pt`에는 SGLang `--speculative-token-map`이 읽는 ID
목록이 저장된다. `--verify` 자료는 포함률을 측정하는 데만 사용하며 어휘 선정에는
쓰지 않는다. `.pt`는 이 도구로 생성한 신뢰할 수 있는 파일을 사용한다.

선정 순서는 byte·특수 토큰과 모든 완성형 한글 포함 토큰을 보존한 뒤 자료의 빈도,
남은 자리는 ID 순서다. 한국어 포함률과 실제 초안 채택률은 다르므로 반드시 전체
어휘 기준과 A/B 비교해야 한다. 모델 가중치와 최종 검증 모델의 어휘는 바꾸지 않는다.


## 재시험 기록

[2026-09-06 3회 재시험](results/2026-09-06-recheck/README.md)은 두 설정의 원본과
시스템 표본을 모두 보존한다. `summarize.py <결과 디렉터리>`로 원본 JSON에서
중앙값·범위·처리량을 다시 계산한다.

`recheck.py`는 이번 로컬 시험에 사용한 실행 기록용 스크립트다. 기존
`sglang-qwen38-fn` 컨테이너의 마운트와 자원 설정 및 `sm121-vocab1` 이미지가 필요하다.
실행 시 컨테이너가 하나라도 떠 있으면 중단하며, 기존 결과 디렉터리에 run.json이
있어도 중단한다. 재사용하려면 출력 디렉터리와 임시 컨테이너 이름을 새로 지정하도록
수정해야 한다. 두 모드를 순서대로 실행하고 각각 3회 측정과 5분 대기를 수행한 뒤
내린다. 재부팅 후 자동 재개하지 않는다. 가용 메모리 8 GiB 미만 표본을 감지하면
시험을 중단하지만 급격한 메모리 고갈을 보장해서 방지하는 장치는 아니다.

결과는 요청마다 임시 파일에 쓰고 fsync 후 교체한다. 장기 보관할 시험 결과에
/tmp 경로를 사용하지 않는다.
