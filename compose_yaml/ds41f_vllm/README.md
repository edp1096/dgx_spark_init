Run DeepSeek-V4.1-Flash on two DGX Sparks (TP2), streaming Engram tables and MoE expert weights from SSD.

## Run

From this directory on the head:

```sh
./manage.sh status
./manage.sh stop
./manage.sh start
./manage.sh logs
```

For more information, see [docs/README.md](docs/README.md).

Source:
* https://github.com/0xBakeer/deepseek-v41-flash-spark
* https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark
* https://x.com/antirez/status/2098121665771110540
* https://github.com/local-inference-lab/b12x
