
* venv
```sh
python3 -m venv my_env
source my_env/bin/activate
```

* 파이썬 hf 설치
```sh
pip install -U "huggingface_hub"
```

* 모델 수동 다운로드
```sh
hf download Qwen/Qwen2.5-Math-1.5B-Instruct --local-dir ./models 
```
