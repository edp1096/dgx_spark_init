# Qwen 3.8 Flash-Next EXL3 3bit · TensorFold

TP1 · Q8 KV · YaRN 4 / 1M · MTP 1–5 (confidence 0.4) · 86,218 draft tokens · vision.

```sh
docker compose up -d --build
docker compose logs -f runtime
```

Uses the existing Huihui checkpoint read-only. Stable parallel expert grouping and compact 16-row tiles improve prefill; one inference prefix is cached. This directory and Talk own separate Compose and build files.

Source: [TensorFold](https://github.com/ashhart/TensorFold/tree/db281878ddb836fd0df510d8771ecb7e0fe47d26), [validation](../../util/talk/docs/qwen38-exl3-tuning-20261021.json).
