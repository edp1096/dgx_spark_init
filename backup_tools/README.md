## docker_image, hf_model

https://github.com/edp1096/file-transfer-sparks 셸스크립트 대신 이거 사용.


## [dgx_recover/make_ventoy_img.sh](./dgx_recover/make_ventoy_img.sh)

* 개요
    * [VenToy](https://www.ventoy.net/en/download.html)로 `DGX Spark`를 위한 `DGX FastOS` 복구이미지(.img)를 생성하고, 이것으로 부팅하여 설치
    * WSL, GX10에서 테스트 (1.120.36)

* 주의사항
    * `VenToy`는 반드시 `ntfs` 또는 `ext4`로 설치. **exfat, fat32 불가**
        * MBR/GPT는 뭘 해도 상관없음
    * 원본 복구이미지와 다르게 TUI 같은거 없이 **아무것도 묻지 않고 바로 복구 진행함**
    * 보안부팅 `Disabled` - Ventoy가 지원을 못함

* 사용법
    -  [DGX Spark User Guide](https://docs.nvidia.com/dgx/dgx-spark/system-recovery.html) 페이지에서 복구 이미지 다운로드 (`dgx-spark-recovery-image-*.tar.gz`)
    - `make_ventoy_img.sh` 파일에서 상단의 `BASE`, `RECOVERY_TAR` 경로 확인 후 실행
    ```sh
    ./make_ventoy_img.sh
    ```
    - 생성된 `.img` 파일을 Ventoy USB(NTFS 또는 ext4)에 복사 후 부팅
