* docker_image, hf_model: https://github.com/edp1096/file-transfer-sparks 셸스크립트 대신 이거 사용.


## dgx_recover/make_ventoy_img.sh

* 개요
    * `VenToy`로 DGX FastOS 복구이미지를 부팅하여 설치
    * WSL에서 실행

* 주의사항
    * `VenToy`는 반드시 `ntfs`로 설치. **exfat, fat32, iso 불가**
        * MBR/GPT는 뭘 해도 상관없음
    * 원본 복구이미지와 다르게 TUI 같은거 없이 **아무것도 묻지 않고 바로 복구 진행함**
    * 보안부팅 `Disabled`

* 사용법

1. [System Recovery](https://docs.nvidia.com/dgx/dgx-spark/system-recovery.html)에서 복구 이미지 다운로드 (`dgx-spark-recovery-image-*.tar.gz`)

2. `make_ventoy_img.sh` 상단의 `BASE`, `RECOVERY_TAR` 경로 확인 후 실행 (tar.gz에서 자동 추출)
```sh
./make_ventoy_img.sh
```

3. 생성된 `.img` 파일을 Ventoy USB(NTFS)에 복사 후 부팅
