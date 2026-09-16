# ModelOpt NVFP4 양자화

`compose.nvfp4.yaml`에서 입력 모델·출력명을, `qwen38_radix_nvfp4.yaml`에서 양자화 설정을 지정한다.

```bash
cd compose_yaml/weights_override/quantization/nvfp4
docker compose -f compose.nvfp4.yaml up --build
```

빌드 컨텍스트는 상위 `weights_override/`이며 공통 `vendor/Model-Optimizer`를 사용한다.
