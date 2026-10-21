# Gemma 4 31B NVFP4 + EXL3 assistant

One DGX Spark, SGLang, BF16 KV, 256K capacity, fixed MTP9, Split-KV verify. Assistant: body4/head6 EXL3. One active request; prefix caching disabled as tested.

```bash
cp -n env.sample .env
docker compose up -d --build
```

Prepare the pinned target and assistant in the Hugging Face cache first. Talk has its own embedded Compose and build files.

Source: [target](https://huggingface.co/lyf/Huihui-gemma-4-31B-it-abliterated-v2-NVFP4), [assistant](https://huggingface.co/edp1096/gemma-4-31B-it-assistant-exl3), [kernel reference](https://github.com/sgl-project/sglang/pull/44830), [measurements](../../util/talk/docs/gemma31-high-mtp-20261021/README.md). Context tested through 64K.
