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

OS 포맷이나 DHCP 변경 후에는 `.env`의 `HEAD_LAN_IP`와
`WORKER_LAN_IP`만 새 주소로 바꾸고, 위 SSH 키 등록과 `manage.sh setup`을
다시 실행한다. QSFP 주소는 `manage.sh`가 매번 복구한다.

DFlash2는 비상업용 라이선스다. 상업용이면 `.env`를 다음과 같이 바꾼다.

```dotenv
SPEC_METHOD=none
MTP_TOKENS=4
```
