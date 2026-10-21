# GLM 5.3 Flash FP4-Local · vLLM TP2

Two DGX Sparks, 1M context, FP8 KV and MTP 5. Huihui abliterated is the default; NVIDIA original is selectable. Both download revisions are pinned.

~~~sh
cp env.sample .env
# Check host addresses, user, cache paths and MODEL_VARIANT.
./manage.sh setup
./manage.sh start
~~~

MTP 1 favors Korean decode. MTP 5 restores code throughput and is the balanced default. A corrected runtime chat template makes thinking switches work without editing checkpoint files. CPU swap is allowed during checkpoint conversion; the 8 GiB KV allocation stays on the GPU.

This standalone recipe and Talk's separately owned recipe_sources/glm53-fp4 are independent deployments. Neither reads the other working directory at runtime.

Source: [NVIDIA FP4-Local](https://huggingface.co/nvidia/GLM-5.3-Flash-FP4-Local), [B12X](https://github.com/lukealonso/b12x), and the comparison/qualification records.

Measured comparison, method and raw results: [qualification record](docs/qualification-20261020/comparison.md). Only the Huihui checkpoint was benchmarked; the selectable NVIDIA original is not separately qualified by these results.
