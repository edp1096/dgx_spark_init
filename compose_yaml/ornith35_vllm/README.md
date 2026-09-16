# Huihui Ornith 1.5 35B · TP1

단일 DGX Spark용 NVFP4·MTP·1M YaRN 구성.

```bash
./manage.sh setup  # 이미지 빌드
./manage.sh start
./manage.sh logs
./manage.sh stop
```

모델: `edp1096/Huihui-Ornith-1.5-35B-A3B-NVFP4`.
빌드·실행 코드는 이 폴더에 있으며 실측 기록은 `docs/`, 검증 도구는 `tools/`에 있다.

출처: [Ornith NVFP4](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-NVFP4), [Huihui BF16](https://huggingface.co/huihui-ai/Huihui-Ornith-1.5-35B-A3B-abliterated).
