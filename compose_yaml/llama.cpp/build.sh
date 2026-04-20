#!/bin/bash

TAG=$(curl -s https://api.github.com/repos/ggml-org/llama.cpp/releases/latest | grep -oP '"tag_name": "\K[^"]+')
TAG+="-v1"
echo "llama.cpp version: $TAG"

echo "Building image..."
docker build -t edp1096/llama.cpp-spark:$TAG . || {
    echo "Build failed."
    exit 1
}

echo "Pushing image..."
docker push edp1096/llama.cpp-spark:$TAG || {
    echo "Push failed. Are you logged in? Run: docker login"
    exit 1
}

echo "Done: edp1096/llama.cpp-spark:$TAG"
