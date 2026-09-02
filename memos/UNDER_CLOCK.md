source: https://forums.developer.nvidia.com/t/cooler-gb10-temps-almost-no-performance-lost/372662

* sudo nano /etc/systemd/system/nvidia-cpu-limit.service
```ini
[Unit]
Description=Set CPU Frequency Limits
After=multi-user.target

[Service]
Type=oneshot
# Use schedutil governor, then cap each cpufreq policy at 70% of hardware max
ExecStart=/usr/bin/bash -c 'cpupower frequency-set -g schedutil; for p in /sys/devices/system/cpu/cpufreq/policy*; do max=$(cat "$p/cpuinfo_max_freq"); echo $((max * 70 / 100)) > "$p/scaling_max_freq"; done && echo "CPU : schedutil governor, clock capped to 70%"'
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
```

* sudo nano /etc/systemd/system/nvidia-gpu-limit.service
```ini
[Unit]
Description=Set NVIDIA GPU Power Limits
After=nvidia-persistenced.service
Wants=nvidia-persistenced.service

[Service]
Type=oneshot
# Enable persistence mode first, wait for GPU to be ready, then set power limits
ExecStart=/usr/bin/bash -c 'sleep 5 && nvidia-smi -pm 1 && nvidia-smi -lgc 0,2100 && echo "GPU : Persistence enabled, Clock limit set to 2100MHz"'
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
```

* sudo systemctl enable nvidia-cpu-limit.service
* sudo systemctl enable nvidia-gpu-limit.service
