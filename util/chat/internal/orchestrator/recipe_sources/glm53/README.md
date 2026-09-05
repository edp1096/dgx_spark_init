# GLM-5.3 Flash — 2× DGX Spark

Head: `192.168.100.61` · Worker: `192.168.100.60` · API: port `8000`

## 최초 1회

Head에서 실행한다.

```bash
test -f ~/.ssh/id_ed25519 || \
  ssh-keygen -t ed25519 -a 64 -f ~/.ssh/id_ed25519 -N ''

ssh-copy-id -i ~/.ssh/id_ed25519.pub edp1096@192.168.100.60

ssh -o BatchMode=yes edp1096@192.168.100.60 hostname

cd /home/edp1096/workspace/dgx_spark_init/compose_yaml/vllm_glm53f
./manage.sh setup
./manage.sh start
```

`setup`은 모델을 Head에 한 번 다운로드하고 Worker로 동기화하며, QSFP
주소(`10.200.0.1`/`10.200.0.2`)와 양쪽 이미지를 준비한다. 중단되면 같은
명령을 다시 실행하면 이어받는다.

## 평소 사용

```bash
./manage.sh start
./manage.sh stop
./manage.sh restart
./manage.sh status
./manage.sh logs
./manage.sh logs worker
```

API 주소는 `http://192.168.100.61:8000/v1`, 모델명은
`glm-5.3-flash`이다.

## 설정

설정 파일은 `.env`다. 파일이 없으면 `manage.sh`가 `env.sample`을 복사해
자동 생성한다. 기본값은 Entrpi `v2.3-tier1`, EXL3 4bpw, DFlash2, 최대
524K context다.

이 SparkTalk 독립 레시피는 Lovesenko `o_proj` 이식을 사용한다.
`ABLIT=1`에서 L0–44의 BF16 가중치 3.5GiB만 Head에 받고 Worker로
동기화한다. MTP는 원본을 유지한다. donor는
`lovesenko/GLM-5.3-Flash-tr3-4bpw-Abliterated`의
`c8f58e6aa9117c73607d692978b22f091d80450c` 리비전이며,
원본 EXL3 및 기존 Dealign 디렉터리는 수정하지 않는다.

OS 포맷이나 DHCP 변경 후에는 `.env`의 `HEAD_LAN_IP`와
`WORKER_LAN_IP`만 새 주소로 바꾸고, 위 SSH 키 등록과 `manage.sh setup`을
다시 실행한다. QSFP 주소는 `manage.sh`가 매번 복구한다.

DFlash2는 비상업용 라이선스다. 상업용이면 `.env`를 다음과 같이 바꾼다.

```dotenv
SPEC_METHOD=none
MTP_TOKENS=4
```

## 공통 관리 명령

`manage.sh setup|image|model|start|stop|restart|status|logs|validate`를 사용한다.
`setup`에 모델 준비가 포함된다. 설정은 `.env`/`env.sample`, 모델 종류는
`MODEL_VARIANT=official|abliterated`이며 `setup`/`model`에 `--official` 또는
`--abliterated`를 지정할 수 있다. HF_TOKEN 환경변수 또는 `--ask-token` 숨김 입력을
사용한다. 상세 규칙은 [공통 CLI](../runtime-common/README.md)를 참조한다.
