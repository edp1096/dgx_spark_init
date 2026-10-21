# Qwen 3.8 Flash-Next FP4-Local · TensorFold

Native CUDA · TP1 · FP8 KV / 1M · MTP 4 · Korean-expanded draft vocabulary · 4K prefill.

```sh
docker compose up -d --build
docker compose logs -f runtime
```

Uses the existing Huihui weights read-only. Supports vision, tools, JSON schema and Thinking off/Medium. Talk has its own embedded Compose and build files.

Source: [TensorFold](https://github.com/ashhart/TensorFold/tree/db281878ddb836fd0df510d8771ecb7e0fe47d26), [recipe](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold), [measurements](../../util/talk/docs/qwen38-fp4-tuning-20261022.json).
