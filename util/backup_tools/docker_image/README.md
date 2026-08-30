
## 셸스크립트

아래 명령들은 참고용이고, 셸스크립트는 gzip 없이 tar만 사용.

* 이미지 압축/저장
```sh
# 압축하지 않는 경우: sudo docker save -o image-name_version.tar image-name:version
sudo docker save nvcr.io/nvidia/vllm:26.01-py3 | gzip > /mnt/ssd_t5/docker_backup/nvcr.io_nvidia_vllm_26.01-py3.tar.gz
```

* 이미지 복원
```sh
# 압축파일 아닌 경우: sudo docker load -i image-name_version.tar
sudo docker load < /mnt/ssd_t5/docker_backup/nvcr.io_nvidia_vllm_26.01-py3.tar.gz
```


## 이미지 전송 #1

* 이미지 추출 - A서버
```sh
docker save -o my_image.tar ${DOCKER_IMAGE}
```

* 파일 전송 - SCP
```sh
scp my_image.tar user@B_SERVER_IP:/path/to/destination
```

* 이미지 복원 - B서버
```sh
docker load -i my_image.tar
```


## 이미지 전송 #2

* ssh 직접 파이프 전송 - A서버
```sh
docker save ${DOCKER_IMAGE} | ssh user@B_SERVER_IP "docker load"
```
