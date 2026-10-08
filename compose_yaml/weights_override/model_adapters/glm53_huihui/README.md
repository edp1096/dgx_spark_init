# GLM-5.3 Flash Huihui GGUF 차이 이식

NVIDIA NVFP4를 기준으로 `DQ(NVIDIA) + DQ(Huihui GGUF) - DQ(Unsloth GGUF)`를 적용하는 후보 변환기다. 변환 결과는 두 DGX Spark에서 실제 1M 입력과 API 회귀검사를 통과했다. 구체적인 조건과 범위는 출력 모델의 `runtime-qualification.json`을 참고한다.

입력 리비전은 작업 매니페스트에 고정한다. GGUF 전체 텐서의 이름·형상·양자화 형식을 검사하고, 원본 해시와 donor를 대조한 뒤 변경 구간만 보관한다. 매핑하지 못한 변경 텐서는 무시하지 않고 실패한다.

- `inspect_headers.py`: HTTP Range로 GGUF 헤더 확인
- `audit_original.py`: 원본 텐서별 해시 생성, 완료 텐서에서 재개
- `compare.py`: donor 전체 비교 및 변경된 원본 구간 취득
- `build.py`: 별도 출력 생성, 원본 SHA256 및 비수정 바이트 검증
- `test_adapter.py`: CPU 수치·매핑·재개 검사

고정밀 텐서는 원래 dtype을 유지하고, NVFP4 텐서는 ModelOpt로 재양자화한다. 활성화 스케일은 보존하며 새 캘리브레이션은 하지 않는다. GGUF 양자화 잔차와 재양자화 오차가 남으므로 Huihui BF16과 동등하거나 원본 성능을 보존한다고 보장하지 않는다. 전체 로딩·추론 및 ablit 효과 검증은 별도 단계다.

현재 자동 작업은 `pipeline.py`로 연결한다. `xet_proof.py`·`proof_ranges.py`에서 동일 내용 블록을 확인하고 미확인 구간을 받는다. `materialize.py`는 재구성한 원본 GGUF의 **파일 전체 SHA256을 Unsloth 공개 값과 대조**한 뒤 텐서 비교로 넘긴다. 구간 증명만으로 원본 검증을 대체하지 않는다. 프로토콜 근거: [Hugging Face Xet](https://huggingface.co/docs/xet/download-protocol).

`download_donor.py huihui` 또는 `download_donor.py nvidia`는 별도 임시 파일·재개 비트맵을 사용한다. 전체 SHA256이 맞은 파일만 HF 캐시에 반영한다. 작업 상태는 `~/.cache/model-download-jobs/glm53-huihui/`에 남긴다. NVFP4는 원본 스케일과 새 스케일 중 복원 오차가 작은 쪽을 선택한다.
