ASUS Ascent GX10(DGX Spark) 초기 설정 기록용


## 목적
* `Docker`, `Python virtual env`를 이용하여 물리환경은 최대한 초기상태 유지.
* [Arcane](https://github.com/getarcaneapp/arcane)을 이용하여 최대한 터미널 타이핑 없는 환경 구성.


## 시작
1. [BEGIN](./memos/0_BEGIN.md) - 초기 설정
1. [ARCANE](./memos/1_ARCANE.md) - Arcane 설치 및 설정
1. [CLUSTER](./memos/2_CLUSTER.md) - 클러스터 설정


## 내용
* [memos](./memos/) - 처음 컴터 켜고 할것들 메모.
* [compose_yaml](./compose_yaml/) - Arcane 외 도커 컴포즈 파일 모음.
* [util](./util/) - 그냥 잡다한 도구.


## 메모
* 재부팅 심한 경우, 부수적으로 소비전력 줄이려는 경우
  - [팬 제어](./util/fan_control) profile 2 또는 3 사용
  - [CPU/GPU 언더클럭](./memos/UNDER_CLOCK.md) - CPU 거버너는 performance 반드시 유지

