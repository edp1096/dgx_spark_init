## 도커 컴포즈 파일

* 메인 - [compose.yaml](../compose_yaml/arcane/compose.yaml)
* 에이전트 - [compose.yaml](../compose_yaml/arcane_agent/compose.yaml)


## 설정

* 그냥 `compose.yaml`만 제대로 놓고 `docker compose up -d` 하면 됨
* 내 경우 1번컴 `~/workspace/arcane` 메인, 2번컴 `~/workspace/arcane_agent` 에이전트로 폴더를 만들어서 각각 넣고 실행함


## 클러스터 워커(에이전트)

### 절차

* 아케인 메인에서 환경 > 환경 추가 > Direct / 적당한 이름 / 아이피:3553(에이전트 포트) > 에이전트 구성 생성
    * IP 뒤에 반드시 Port(3553) 적어야 함
