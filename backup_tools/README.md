* docker_image, hf_model: https://github.com/edp1096/file-transfer-sparks 셸스크립트 대신 이거 사용.


## dgx_recover/main_ventoy_iso.sh

* 개요
    * wsl에서 테스트
    * USB는 E드라이브로 가정하며 USB 드라이브를 쓰지 않고 ssd로 복사하여 작업

* [System Recovery](https://docs.nvidia.com/dgx/dgx-spark/system-recovery.html) 내용대로 복구 usb 생성
* wsl에서 usb드라이브 마운트 및 테스트
```sh
sudo mkdir -p /mnt/e
sudo mount -t drvfs E: /mnt/e
ls /mnt/e/
sudo umount /mnt/e
```
* cmd에서 복구usb ssd로 복사
    * 아래 첫 줄 cd는 복사할 위치로 변경. **chkdsk 실행시 /mnt/e 마운트 해제됨**
```cmd
cd D:\dev\asus_ascent_gx10_dgx_spark\dgx_usb

chkdsk E: /F
xcopy E:\* .\ /E /H
```
* iso 생성 스크립트 내용 상단의 **경로 수정 후** 실행
```sh
./make_ventoy_iso.sh
```
