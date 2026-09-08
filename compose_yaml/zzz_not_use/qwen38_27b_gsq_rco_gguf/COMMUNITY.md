# RentedNoodle 후보 확인 — 2026-09-08

사용자가 제시한 [모델](https://huggingface.co/RentedNoodle/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-Uncensored)의
고정 리비전은 `0fe4ddcc59be3db84402569eb16b5d66f5ad1cff`이다.
README·배분표·평가 명령·공개 런타임 양자화 코드를 확인했다. 모델 가중치를 다운로드하거나
실행한 검증은 아직 하지 않았다.

- 입력은 우리가 가진 것과 같은 Huihui-Qwen3.8-27B-abliterated 모델 계열이다.
  제작에 사용한 정확한 입력 리비전은 모델 카드에서 확인되지 않았다.
- 약 10.44 GB의 IQ3_XXS 크기급 혼합 정밀도 GGUF이며 MTP를 포함한다고 명시한다.
- 제작자는 ISTA의 최종 per-tensor allocation과 imatrix를 재사용했으며 RCO 탐색을
  독립 실행하지 않았다고 명시한다.
- 공개 재현 설명은 `llama-quantize --tensor-type-file`이다. 해당 옵션은 텐서별
  양자화 타입을 지정한다. 이 정보만으로 GSQ가 학습한 코드/scale을 재현하지는 않는다.
- 명시된 런타임은 `RentedNoodle/den_llama.cpp`의
  `3231ee89dbb6db361e2f87fe1bf9fa7db73ad0b6`이다. 확인한 `tools/quantize/quantize.cpp`,
  `src/llama-quant.cpp`, `ggml/src/ggml-quants.c`에서 공개 경로는 일반 타입 선택과
  `ggml_quantize_chunk` 호출이다. GSQ 최적화를 별도 수행한 근거는 확인되지 않았다.

따라서 이 후보는 **ISTA 배분 재사용 기반 Huihui 혼합 양자화**로 구분해 평가한다.
새 Huihui 후보를 GSQ로 최적화하고 RCO를 재실행하는 README의 원래 목표와는 다르다.
후자의 전체 경로 미확보 상태는 유지되지만, 전자를 별도 실험으로 만드는 것은
공개 구성요소를 활용할 수 있는 대안이다. IQ3_S 배분으로의 확장도 기술적으로 검토할 수
있으며 결과 품질과 크기는 새로 검증해야 한다.

제작자의 속도는 RTX 5070 Ti/Windows/별도 llama.cpp 빌드 조건이다. MTP n=2의
개선과 n=3/4의 저하는 Spark 실측을 대신하지 않는다. 도구 호출 v1 8/8과 v2 7/14의
평가 조건도 다르다. `Livebench-style`은 공식 LiveBench 점수가 아닌 자체 문항 평가다.
MTP 재학습 v1.1은 진행 중으로 표시되며 배포 완료로 취급하지 않는다.

확인한 파일:

- [제작 설명](https://huggingface.co/RentedNoodle/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-Uncensored/blob/0fe4ddcc59be3db84402569eb16b5d66f5ad1cff/README.md)
- [배분표](https://huggingface.co/RentedNoodle/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-Uncensored/blob/0fe4ddcc59be3db84402569eb16b5d66f5ad1cff/REF-IQ3_XXS-mtp.rco-allocation.txt)
- [평가 명령](https://huggingface.co/RentedNoodle/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-Uncensored/blob/0fe4ddcc59be3db84402569eb16b5d66f5ad1cff/evals/commands.md)
- [양자화 CLI](https://github.com/RentedNoodle/den_llama.cpp/blob/3231ee89dbb6db361e2f87fe1bf9fa7db73ad0b6/tools/quantize/quantize.cpp)
