## 클러스터 설정

* Source: https://github.com/nvidia/dgx-spark-playbooks/tree/main/nvidia/connect-two-sparks
* username은 둘다 edp1096으로 간주

* 물리 연결 확인
    * `enp1s0f0np0`와 `enp1s0f1np1` 사용
    * `enP2p1s0f0np0`와 `enP2p1s0f1np1`는 alias니까 무시하랜다.
```sh
ibdev2netdev

#port 1 ==> enP2p1s0f0np0 (Up)
#port 1 ==> enP2p1s0f1np1 (Down)
#port 1 ==> enp1s0f0np0 (Up)
#port 1 ==> enp1s0f1np1 (Down)

ifconfig enp1s0f0np0
#enp1s0f0np0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
#        inet 169.254.14.7  netmask 255.255.0.0  broadcast 169.254.255.255
# ...
#
```

* netplan 설정 - SQFP daisy-chain
```sh
sudo tee /etc/netplan/40-cx7.yaml > /dev/null <<EOF
network:
  version: 2
  ethernets:
    enp1s0f0np0:
      link-local: [ ipv4 ]
    enp1s0f1np1:
      link-local: [ ipv4 ]
EOF

sudo chmod 600 /etc/netplan/40-cx7.yaml
sudo netplan apply
```

* avahi로 spark 찾아서 자동로그인 설정
```sh
wget https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/refs/heads/main/nvidia/connect-two-sparks/assets/discover-sparks

chmod +x ./discover-sparks
./discover-sparks

ifconfig enp1s0f0np0
```

* 롤백 - vllm ray cluster 계속 쓸거니까 필요 없음.
```sh
sudo rm /etc/netplan/40-cx7.yaml
sudo netplan apply
```
