#!/bin/bash

# export DOCKER_USERNAME=edp1096
# export DOCKER_TOKEN=dckr_pat_xxxxxxxxxxxxxxxxx
# chmod +x docker-login.sh
# ./docker-login.sh



USERNAME="edp1096"
TOKEN=""

DOCKER_USERNAME="${DOCKER_USERNAME:-$USERNAME}"
DOCKER_TOKEN="${DOCKER_TOKEN:-$TOKEN}"

if [ -z "$DOCKER_USERNAME" ] || [ -z "$DOCKER_TOKEN" ]; then
    echo "Error: Docker credentials not set"
    echo "Method 1: Set environment variables"
    echo "  export DOCKER_USERNAME=your_username"
    echo "  export DOCKER_TOKEN=your_access_token"
    echo "Method 2: Edit this script and set USERNAME and TOKEN variables"
    echo "Create token at: https://hub.docker.com/settings/security"
    exit 1
fi

echo "Logging in to Docker Hub..."
echo "$DOCKER_TOKEN" | docker login -u "$DOCKER_USERNAME" --password-stdin

if [ $? -eq 0 ]; then
    echo "Login successful"
else
    echo "Login failed"
    exit 1
fi