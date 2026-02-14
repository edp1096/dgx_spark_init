## 실행

* `edp1096/ace-step-spark` 이미지로 실행
```sh
docker pull edp1096/ace-step-spark
docker run --gpus all -it --rm --network host edp1096/ace-step-spark
docker exec -it <container_id> "cd ~/play/ACE-Step-1.5/ && ./start_gradio_ui.sh"
```

* `nvidia/pytorch` 이미지에서 실행
```sh
docker pull nvidia/pytorch:26.01-py3
docker run --gpus all -it --rm --network host nvidia/pytorch:26.01-py3
docker exec -it <container_id> bash

# copy run_acestep15.sh into the container and run
docker mkdir /workspace
docker cp ./run_acestep15.sh <container_id>:/workspace/run_acestep15.sh
docker exec -it <container_id> bash -c "cd /workspace && ./run_acestep15.sh"
```


## torchcodec 직접 빌드

* 도커 먼저 올리고 `build_torchcodec.sh` 스크립트 복사 후 실행
```sh
docker pull nvidia/pytorch:26.01-py3
docker run --gpus all -it --rm --network host nvidia/pytorch:26.01-py3
docker exec -it <container_id> bash
# copy build_torchcodec.sh into the container and run
docker mkdir /workspace
docker cp ./build_torchcodec.sh <container_id>:/workspace/build_torchcodec.sh
docker exec -it <container_id> bash -c "cd /workspace && ./build_torchcodec.sh"
```
