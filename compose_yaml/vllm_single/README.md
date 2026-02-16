## compose.yaml 다른거

* Claude code 연결용
```yaml
services:
  vllm:
    image: "nvcr.io/nvidia/vllm:26.01-py3"
    container_name: vllm-single
    network_mode: "host"
    ipc: "host"
    volumes:
      - "/home/edp1096/workspace/models:/root/.cache/huggingface"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    environment:
      - TORCH_NCCL_ASYNC_ERROR_HANDLING=1
      - VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
      # - HF_TOKEN=YOUR_HF_TOKEN_HERE
    command: >
      python3 -m vllm.entrypoints.openai.api_server
      --host 0.0.0.0
      --port 8000
      --max-model-len 2048
      --gpu-memory-utilization 0.8
      --max-num-seqs 256
      --disable-log-requests
      --trust-remote-code
      --enforce-eager
      --max-model-len 128000
      --served-model-name my-model --enable-auto-tool-choice --tool-call-parser openai
      --model openai/gpt-oss-120b
    # --model nvidia/Qwen3-8B-NVFP4
    # --model Qwen/Qwen2.5-Math-1.5B-Instruct
    # --model nvidia/Llama-3.3-70B-Instruct-NVFP4
    # --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8
```