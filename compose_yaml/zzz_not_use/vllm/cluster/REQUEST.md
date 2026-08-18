```sh
curl http://192.168.100.60:8000/v1/chat/completions -H "Content-Type: application/json" -d '{ 
    "model": "nvidia/Llama-3.3-70B-Instruct-NVFP4",
    "messages": [
      {"role": "user", "content": "안녕? 너는 누구야? DGX Spark에서 잘 돌아가고 있어?"}
    ]
  }'
```

```sh
curl http://192.168.100.60:8000/v1/chat/completions -H "Content-Type: application/json" -d '{ 
    "model": "nvidia/Llama-3.3-70B-Instruct-NVFP4",
    "messages": [
      {"role": "user", "content": "Write a haiku about a GPU"}
    ]
  }'
```

```sh
curl http://192.168.100.60:8000/v1/chat/completions -H "Content-Type: application/json" -d '{ 
    "model": "nvidia/Llama-3.3-70B-Instruct-NVFP4",
    "messages": [
      {"role": "user", "content": "Write a haiku about a GPU. 한국말로 답해라."}
    ]
  }'
```

```sh
curl http://192.168.100.60:8000/health
```