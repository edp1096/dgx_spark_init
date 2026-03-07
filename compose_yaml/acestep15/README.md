# ACE-Step 1.5 for DGX Spark

* Based on
  * [NGC Pytorch 26.02-py3](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch?version=26.02-py3)
  * [ACE-Step1.5 v0.1.4](https://github.com/ace-step/ACE-Step-1.5/releases/tag/v0.1.4)
* Mods
  * Flash attention 2 enabled
  * Add Korean to i18n
* Docker image: https://hub.docker.com/repository/docker/edp1096/ace-step-spark


## Run

* Docker Run
```sh
docker run --gpus all -it --rm \
  -p 7860:7860 \
  -v /home/edp1096/.cache/huggingface:/root/.cache/huggingface \
  edp1096/ace-step-spark:1.5-arm64 \
  bash -c "cd ~/ACE-Step-1.5 && acestep --port 7860 --server-name 0.0.0.0 --config_path acestep-v15-turbo --lm_model_path acestep-5Hz-lm-0.6B --init_service true --offload_to_cpu true"
```

* Docker Compose - See [compose.yaml](./compose.yaml)
```sh
docker compose up -d
```


## Build

See [build_acestep.sh](builder/build_acestep.sh)
