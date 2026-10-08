# GLM NVFP4 가중치 복구

Talk의 GLM TP2는 SGLang+DFlash2로 실행하며, NVIDIA NVFP4 또는 로컬에서 변환한 Huihui NVFP4를 사용합니다.
기존 EXL3 본체와 EXL3용 DFlash 파일은 사용하지 않습니다.

1. GLM 모델세트를 중지합니다.
2. **설정 → 시스템 → 모델 준비**에서 GLM과 가중치 종류를 선택합니다.
3. **모델만 준비**로 헤드의 파일을 확인하고 워커에 동기화한 뒤 세트를 시작합니다.

NVIDIA 원본과 Huihui 공개 모델 모두 검증된 리비전으로 다운로드합니다.
Huihui는 `edp1096/Huihui-GLM-5.3-Flash-abliterated-NVFP4`이며, 완성된 로컬 변환 결과가 있으면 그대로 사용합니다.

터미널에서는 실행 디렉터리의 `./manage.sh model --official` 또는
`./manage.sh model --abliterated`로 준비하고 `./manage.sh start`로 시작합니다.
`./manage.sh image`는 헤드의 런타임 이미지를 워커와 일치시킵니다.

시작 전 검사는 모델 설정·양자화 형식·33개 shard의 헤더와 파일 길이·인덱스의
텐서 존재 여부를 확인합니다. 빠른 구조 검사이므로 전체 SHA-256 검사를 대신하지 않습니다.
실행 설정과 검증 범위는 `compose_yaml/glm53f_sglang/README.md`와 `qualification.json`에 기록합니다.
