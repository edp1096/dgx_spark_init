* Node B (gx10-f05a) - 수신
```sh
# 인터페이스 확인
ibdev2netdev

# 두 번째 인터페이스 IP 및 MTU
sudo ip addr add 169.254.45.1/16 dev enP2p1s0f0np0
sudo ip link set enP2p1s0f0np0 mtu 9000
sudo ip link set enP2p1s0f0np0 up

# 수신 대기
nc -l -p 9998 > /tmp/part1 &
nc -l -p 9999 > /tmp/part2 &
wait
cat /tmp/part1 /tmp/part2 > /dest/file
```

* Node A (gx10-bb75) - 송신
```sh
# 인터페이스 확인
ibdev2netdev

# 두 번째 인터페이스 IP 및 MTU
sudo ip addr add 169.254.142.1/16 dev enP2p1s0f0np0
sudo ip link set enP2p1s0f0np0 mtu 9000
sudo ip link set enP2p1s0f0np0 up

# 파일 분할 후 송신
split -n 2 /source/file /tmp/split_
nc -s 169.254.141.58 169.254.44.172 9998 < /tmp/split_aa &
nc -s 169.254.142.1 169.254.45.1 9999 < /tmp/split_ab &
wait
rm /tmp/split_aa /tmp/split_ab
```



DGX Spark Connectx-7 Infiniband 연결은 NCCL같은 특수목적용이 아닌 단순 파일 전송에서는 그냥 100GbE라고 간주해야겠다.

Source:
* https://forums.developer.nvidia.com/t/connectx-7-nic-in-dgx-spark/350417
* https://www.servethehome.com/the-nvidia-gb10-connectx-7-200gbe-networking-is-really-different