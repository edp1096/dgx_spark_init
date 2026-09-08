> 보관된 구성입니다. SparkTalk 메인은 [Qwen 27B EXL3](../../qwen38_27b_exl3/README.md)로 복귀했습니다. 아래 성능·검사 기록은 기존 GGUF 구성의 기록입니다.

# Qwen3.8 27B · GSQ-RCO 배분표 기반 GGUF

Huihui BF16 원본에 ISTA의 IQ3_S-MTP 텐서별 정밀도 배분표와 imatrix를 적용해
혼합 정밀도 GGUF를 만들고 단일 DGX Spark에서 실행하는 독립 폴더다.

**GSQ 가중치 최적화나 RCO 탐색을 새로 수행한 모델은 아니다.**
방법과 고정 리비전은 `settings.json`, 조사 근거는 [COMMUNITY.md](COMMUNITY.md)에 있다.
공식 ISTA 모델의 벤치마크 점수를 이 결과물에 적용하지 않는다.

## 준비 및 제작

호스트에는 Docker Compose와 NVIDIA Container Toolkit/드라이버가 필요하다.
Python·PyTorch·변환기·llama.cpp·CUDA 빌드는 Dockerfile에서 준비한다.
다른 런타임 폴더의 스크립트나 Python 환경을 참조하지 않는다.

```sh
cd /home/edp1096/workspace/dgx_spark_init/compose_yaml/qwen38_27b_gsq_rco_gguf
cp env.sample .env                 # 설정을 바꿀 때만 필요
./manage.sh setup                  # 이미지 빌드 → 원본 준비 → 변환 → 양자화 → 구조 검사
```

처음에는 고정된 Huihui 원본을 다운로드한다. 이미 다운로드한 HF snapshot을 사용하려면
다음처럼 **복사해 가져온 뒤** 제작한다. 가져오기 완료 후 기존 캐시 경로는 필요 없다.

```sh
./manage.sh tools-image
./manage.sh import-source /absolute/path/to/Huihui-BF16/snapshot
./manage.sh image
./manage.sh quantize
```

원본 shard SHA256을 고정 HF 리비전과 대조한다. 참조 배분표/imatrix도 SHA256을 확인한다.
변환된 텐서 이름 866개, 각 최종 저장 타입, MTP 15개가 배분표와 일치해야 성공한다.
중간 결과는 유지되어 완료한 단계부터 재사용한다. 최초 준비에는 다운로드 캐시와
중간 BF16 파일을 포함해 약 250GB의 여유 디스크를 권장한다.

`converter.py`는 변환기의 임시 행 결과를 순차 처리해 메모리 중복을 줄인다.
같은 BF16 바이트가 나오는지 `test_converter.py`에서 검사한다.

## 실행

```sh
./manage.sh start
./manage.sh status
./manage.sh logs
./manage.sh stop
```

기본 32K 프로필은 시작 전에 가용 메모리 24GiB 이상을 확인한다.

기본 API는 `http://127.0.0.1:18696/v1`, 모델 별칭은 `qwen3.8-27b-ista-iq3s-mtp`이다.
기본값은 32K 문맥, Q8 K/V cache, MTP 2단계, BF16 vision projector이다.
설정은 `.env`의 API_PORT, CONTEXT_SIZE, SPEC_TYPE, MTP_TOKENS에서 조정한다.
MTP를 끄려면 SPEC_TYPE=none으로 설정한다. 컨텍스트 변경 시 필요한 메모리도 달라진다.

이 서버는 기존 SparkTalk 모델의 설정을 자동으로 바꾸지 않는다.

## 검증

```sh
./manage.sh verify                # 구조·정밀도 배분·MTP·체크섬
./manage.sh validate --quality    # 이미 실행 중인 기본 포트 서버의 응답/도구/비전 검사
./benchmark.sh                   # serial/MTP 1/2/3 비교, 완료 시 시험 서버 정지
```

포트를 바꿨다면 validate에 `--url http://127.0.0.1:새포트`를 전달한다.
다른 LLM이 메모리를 점유하는 환경에서는 시험 동안만 그 컨테이너를 정지·복구하는
명시적 옵션이 있다. 해당 모델을 사용하는 요청이 없는 상태에서 실행한다.

```sh
./benchmark.sh --pause-container EXISTING_LLM_CONTAINER --resume-url http://127.0.0.1:8000/v1/models
```

스크립트는 정상 종료·실패·중단 시 시험 서버를 정지하고, 자신이 정지했던 기존
컨테이너를 다시 시작해 API 응답을 최대 15분 기다린다.
포트가 자동 판별되지 않는 런타임에는 `--resume-url`을 지정한다. 운영 모델을 자동 교체하는 명령은 아니다.

속도 비교는 기존 4개 프롬프트(영어 코드/수학, 한국어 기술/산문), thinking off,
최대 출력 512토큰, temperature 0.6, seed 42를 사용한다. client 처리량과 서버 decode
timing을 따로 기록한다. 도구·이미지·검색·한국어 검사는 소규모 동작 검사이며
공인 벤치마크나 모든 요청에서의 비거부 동작을 보장하지 않는다.

## 산출물

모든 입력·중간 파일·모델·검사 결과는 이 폴더의 `data/`에 있다.

- `source/`: Huihui BF16 원본과 원본 체크섬 기록.
- `reference/`: 고정 ISTA 배분표, imatrix, 정규식 이름을 정확히 매칭하는 quantizer 입력.
- `work/source-bf16-mtp.gguf`: MTP 포함 고정밀 중간 파일.
- `models/Huihui-Qwen3.8-27B-ISTA-IQ3_S-Allocation-MTP.gguf`: 약 12.12GB 최종 모델.
- `models/mmproj-Huihui-Qwen3.8-27B-BF16.gguf`: 약 0.93GB 비전 인코더/projector.
- `logs/`: 각 제작 단계의 명령·시간·출력.
- `reports/model.json`: 구조·정밀도 배분·체크섬·제작 출처.
- `reports/{serial,mtp1,mtp2,mtp3}/`: 실제 요청·응답·속도·검사 결과.

`data/`는 Git에서 제외한다. 폴더를 다른 장비로 옮길 때 모델도 재사용하려면
`data/`를 함께 복사한다. Docker 이미지는 그 장비에서 `./manage.sh image`로 만든다.

## 이번 생성 결과

2026-09-08 생성·실행 검증 결과는 [EVALUATION.md](EVALUATION.md)에 있다.
MTP 2단계 23.79 tok/s, 3단계 25.14 tok/s를 측정했다. 사실 오류도 발견되어
추가 비교 후 SparkTalk 메인 모델로 선정했다. 앱 내장 프로필은 MTP 3·캐시 1GiB를 사용한다.

NVFP4+DFlash2, EXL3+MTP와 같은 조건에서 다시 측정한 결과는
[COMPARISON-27B.md](COMPARISON-27B.md)에 있다. GGUF MTP 3과 프롬프트 캐시
1GiB 조합은 24.80 tok/s, 실행 후 추가 메모리 약 17.3GiB를 기록했다.
이 캐시는 32K KV 문맥 한도와 별도다. 이 조합의 실행 옵션은
`MTP_TOKENS=3 CACHE_RAM_MIB=1024 ./manage.sh start`로 지정한다.
정확한 전체 비교 재현 명령과 원본 응답 위치는 [compare/README.md](compare/README.md)를 참고한다.

SparkTalk 메인 적용과 후속 128K 검증은 [DEPLOYMENT.md](DEPLOYMENT.md)에 기록했다.
