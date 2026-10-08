* compose.yaml
```yaml
services:
  open-webui:
    image: ghcr.io/open-webui/open-webui:ollama
    container_name: open-webui
    ports:
      - "12000:8080"
    volumes:
      - open-webui:/app/backend/data
      - open-webui-ollama:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

volumes:
  open-webui:
  open-webui-ollama:
```

* volume 위와 같이 임의 지정인 경우 위치
```sh
/var/lib/docker/volumes/open-webui_open-webui/_data
```
