ASUS Ascent GX10(DGX Spark) 초기 설정 기록용


## 목적

* `Docker`, `Python virtual env`를 이용하여 물리환경은 최대한 초기상태 유지.
* [Arcane](https://github.com/getarcaneapp/arcane)을 이용하여 최대한 터미널 타이핑 없는 환경 구성.


## 시작

1. [BEGIN](./memos/0_BEGIN.md) - 초기 설정
1. [ARCANE](./memos/1_ARCANE.md) - Arcane 설치 및 설정
1. [CLUSTER](./memos/2_CLUSTER.md) - 클러스터 설정
1. 새로 만드는 이미지는 [pytorch_base](./compose_yaml/pytorch_base/) 기반으로 생성. 예: [ACE-Step1.5](./compose_yaml/acestep15/), NVIDIA 플레이북의 [CompfyUI](https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/comfy-ui) 등.

## 내용

* [memos](./memos/) - 처음 컴터 켜고 할것들 메모.
* [backup_tools](./backup_tools/) - 도커 이미지 및 허깅페이스 모델 백업/복원 도구.
* [compose_yaml](./compose_yaml/) - Arcane 외 도커 컴포즈 파일 모음.
