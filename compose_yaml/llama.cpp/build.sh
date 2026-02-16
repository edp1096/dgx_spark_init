#!/bin/bash

if ! docker info 2>/dev/null | grep -q "Username:"; then
    echo "Error: Not logged in to Docker Hub"
    echo "Please run: docker login"
    exit 1
fi

TAG=$(curl -s https://api.github.com/repos/ggml-org/llama.cpp/releases/latest | grep -oP '"tag_name": "\K[^"]+')
TAG+="-v1"
echo "llama.cpp version: $TAG"


cd builder
docker compose up -d
echo "Waiting for build:"

ELAPSED=0
while true; do
    if docker logs llama-cpp-spark 2>&1 | grep -q "====== READY FOR COMMIT ======"; then
        echo "Runtime container is ready!"
        break
    fi
    echo "Building. passed second: $ELAPSED sec"
    sleep 20
    ELAPSED=$((ELAPSED + 20))
done

echo "Commit and compose down:"
docker commit llama-cpp-spark edp1096/llama.cpp-spark:$TAG
docker compose down -v

echo "Push:"
docker push edp1096/llama.cpp-spark:$TAG

echo "Done."
