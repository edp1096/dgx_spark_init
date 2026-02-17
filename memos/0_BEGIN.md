## 업데이트, 도커 실행 등

* 업데이트
```sh
sudo apt update
sudo apt upgrade
```

* 비허용 패턴으로 비번 변경
```sh
sudo su
passwd edp1096
```

* 도커 사용자 그룹 추가 - sudo 안쓰게 하기
```sh
sudo usermod -aG docker $USER
newgrp docker
```


## DGX-OS 재설치한 경우 대용량 파일 처리

* `backup_tools`로 도커 이미지와 허깅페이스(hf) 모델 백업 및 복원
* 모델은 가급적 자동 말고 수동으로 다운로드
* hf 모델 다운로드시 폴더명은 `hf_models/hub` 밑에 `models--account--model_name-params` 식으로 생성.
    * -- = /
