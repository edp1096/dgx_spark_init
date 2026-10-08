# dealignai와 Huihui-RadixArk 가중치 직접 비교

대상은 dealignai revision `be794b990578ef3031eccf9f28e675a289a09ee9`와 공개한 edp1096 모델 revision `40f09f531da577a4fcfbd6b368e7b6ebacde403e`의 로컬 체크포인트다. RadixArk revision `7b719225242aacd3dbd3f9407468c2ee9a9d2594`를 공통 기준으로 사용했다.

실행 중인 모델을 내리지 않고 모든 safetensors 항목을 직접 비교했다. 같은 inode를 공유하는 파일은 동일 파일임을 확인했고, 나머지는 전체 텐서 데이터 SHA256을 계산했다. BF16 텐서는 원본 대비 변경량과 두 모델 간 수치 차이도 계산했다. 표본 추출이 아니다.

- 전체 206개 샤드 중 182개 동일, 24개 다름
- 전체 296,475개 텐서 중 290,194개 동일, 6,281개 다름
- 다른 텐서 수에는 NVFP4 packed weight와 각 scale 텐서가 따로 포함된다. 이 비율을 모델 품질·파라미터 비율로 해석하지 않는다.

| 공통 RadixArk 원본 대비 | dealignai | 새 모델 |
|---|---|---|
| 일반 어텐션 출력 12개 | 변경 | 변경, 서로 다른 값 |
| GDN 출력 36개 | 원본 유지 | 변경 |
| shared expert down 48개 | 원본 유지 | 변경 |
| routed expert down (2·4·30·46·47번 레이어) | 원본 유지 | 2,560 experts 변경 |
| MTP attention output 1개 | 변경 | 원본 유지 |
| 기타 가중치 | 동일 | 동일 |

직접 달라진 텐서는 routed-expert packed weight 2,560개, FP8 block scale 2,560개, FP32 global scale 1,064개, BF16 텐서 97개다. BF16 97개는 본체 96개와 MTP 1개다.

`config.json`, `hf_quant_config.json`, `generation_config.json`, `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, `preprocessor_config.json`, `model.safetensors.index.json`은 바이트 단위로 동일하다.

두 모델에서 모두 수정된 일반 어텐션 출력 12개의 원본 대비 변경 벡터 코사인 유사도 중앙값은 약 0.0051이다(범위 -0.00037~0.02498). 동일 변경량에 작은 반올림 차이만 붙은 관계로 보기 어렵다. 이 값은 텐서 변경 방향의 수학적 유사도이며 답변·의미·품질 유사도가 아니다.

두 모델 간 BF16 상대 L2 차이의 텐서별 중앙값은 GDN 3.91%, shared expert 4.00%, 일반 어텐션 18.65%다. MTP 출력은 12.68% 다르다. 이것도 가중치 수치 차이이며 정답률 저하율이 아니다.

결론: 두 모델은 같은 기반을 많이 공유하지만 동일 모델이 아니며, 수정 범위와 수정값이 모두 다르다. 이번 작업은 가중치·설정의 직접 비교이고, 답변 A/B 비교는 아니다.

원시 결과: [dealign-comparison.json](dealign-comparison.json). 도구: `../compare_dealign.py`. 실행 시간 약 118초.
