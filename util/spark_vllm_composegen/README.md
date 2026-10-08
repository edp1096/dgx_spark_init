# spark-vllm-composegen

`eugr/spark-vllm-docker` 스타일의 vLLM 실행 설정을 Docker Compose YAML로 출력하는 작은 Go 도구.

목표는 `launch-cluster.sh exec vllm serve ...`로 검증한 옵션을 repo 안의 compose 파일로 고정하기 쉽게 만드는 것이다. 입력도 eugr recipe와 같은 YAML을 쓴다.

## 사용

```sh
cd spark_vllm_composegen
go run . -config examples/gemma4-bg-26b-single.yaml
```

파일로 저장:

```sh
go run . -config examples/gemma4-bg-26b-single.yaml \
  -out ../compose_yaml/vllm_gemma4/single/compose.generated.yaml
```

Ray cluster head/worker compose 생성:

```sh
go run . -config examples/gemma4-bg-26b-head.yaml -out head.compose.yaml
go run . -config examples/gemma4-bg-26b-worker.yaml -out worker.compose.yaml
```

eugr recipe-style 입력도 받을 수 있다:

```sh
go run . -config examples/gemma4-bg-26b-recipe-ray-head.yaml
```

No-Ray multi-node compose 생성:

```sh
go run . -config examples/gemma4-bg-26b-noray-head.yaml -out head.noray.compose.yaml
go run . -config examples/gemma4-bg-26b-noray-worker.yaml -out worker.noray.compose.yaml
```

## 지원 범위

- `single`, `ray-head`, `ray-worker`, `noray-head`, `noray-worker` compose 생성
- `head`, `worker`는 호환 alias이며 각각 `ray-head`, `ray-worker`로 처리
- host network, GPU reservation, Hugging Face cache mount
- eugr 기본 Ray/NCCL/UCX env 주입
- Ray head/worker bootstrap command 생성
- No-Ray 모드에서 `--distributed-executor-backend ray` 제거 후 `--nnodes`, `--node-rank`, `--master-addr`, `--master-port`, `--headless` 자동 추가
- `vllm serve` 모델과 인자 배열을 compose command로 출력
- eugr recipe-style `recipe_version`, `name`, `description`, `model`, `container`, `build_args`, `mods`, `defaults`, `env`, `cluster_only`, `solo_only`, `command` 입력
- `command`의 `{defaults}` placeholder 치환 후 `vllm serve <model>` 명령을 파싱
- `pre_commands`로 패치 복사 같은 시작 전 작업 삽입

eugr recipe를 compose 대상으로 해석하지만, `--setup`, build/copy, model download/copy, autodiscovery는 실행하지 않는다. 이 도구는 실행 가능한 compose YAML 생성만 담당한다.

## Recipe 입력

eugr recipe 필드는 그대로 둘 수 있고, compose 생성에 필요한 필드를 추가한다.

```yaml
recipe_version: "1"
name: My Model
model: org/model
container: vllm-node-tf5
build_args:
  - --tf5
mods:
  - mods/my-fix
defaults:
  port: 8000
  host: 0.0.0.0
command: |
  vllm serve org/model \
    --host {host} \
    --port {port}

mode: ray-head
head_ip: 169.254.141.58
node_ip: 169.254.141.58
```

처리 방식:

- `container`는 compose의 `image`로 변환한다.
- `defaults`는 `command`의 `{name}` placeholder에 치환한다.
- `command`에서 `vllm serve <model>`을 찾아 `model`과 인자로 분리한다.
- `cluster_only: true`는 `single` mode 생성을 막는다.
- `solo_only: true`는 cluster mode 생성을 막는다.
- `build_args`와 `mods`는 읽어서 허용하지만, compose 출력에서 자동 실행하지 않는다.

## Ray / No-Ray 모드

`ray-head`는 compose command 안에서 Ray head를 먼저 시작한 뒤 `vllm serve ... --distributed-executor-backend ray`를 실행한다. `ray-worker`는 `ray start --address=HEAD:PORT --block`만 실행한다.

`noray-head`와 `noray-worker`는 Ray를 시작하지 않는다. 대신 같은 `vllm serve` 명령을 각 노드에서 실행하고, vLLM native distributed 인자를 붙인다. worker에는 `--headless`가 추가된다.

No-Ray 모드에서 필요한 필드:

- `head_ip`
- `num_nodes`
- `node_rank`
- `master_port`

## mods 처리

`spark-vllm-docker`의 `--apply-mod mods/...`는 현재 자동 변환하지 않는다. 원본 스크립트는 컨테이너를 먼저 띄운 뒤 `mods/<name>/run.sh`를 `docker cp`로 복사하고 컨테이너 내부에서 실행한다. Compose 파일 하나로 같은 동작을 정확히 재현하려면 mod별 파일 마운트 위치와 실행 순서를 알아야 한다.

그래서 이 도구에서는 필요한 mod 작업을 YAML의 `pre_commands`에 명시적으로 넣는다. 예를 들어 Gemma4 26B의 `gemma4_patched.py` 덮어쓰기는 `pre_commands`와 `volumes`로 표현한다. 여러 파일을 패치하거나 `run.sh`가 복잡한 mod는 compose 생성 전에 수동으로 확인해야 한다.
