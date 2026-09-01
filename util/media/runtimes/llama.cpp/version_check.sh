#!/bin/bash

# Extract prefix (letter) and number from tag
extract_version() {
    local tag="$1"
    # Remove -v[number] suffix if exists
    tag=$(echo "$tag" | sed 's/-v[0-9]\+$//')
    local prefix=$(echo "$tag" | sed 's/\([a-z]\+\).*/\1/')
    local number=$(echo "$tag" | sed 's/[a-z]\+\([0-9]\+\)/\1/')
    echo "$prefix $number"
}

# Compare two tags
compare_tags() {
    local llama_tag="$1"
    local current_tag="$2"
    
    read -r llama_prefix llama_num <<< $(extract_version "$llama_tag")
    read -r current_prefix current_num <<< $(extract_version "$current_tag")
    
    if [[ "$llama_prefix" > "$current_prefix" ]]; then
        echo "new_release"
    elif [[ "$llama_prefix" < "$current_prefix" ]]; then
        echo "rollback"
    else
        if (( llama_num > current_num )); then
            echo "new_release"
        elif (( llama_num < current_num )); then
            echo "rollback"
        else
            echo "same"
        fi
    fi
}

# Get llama.cpp latest tag
echo "Checking llama.cpp latest version..."
llama_tag=$(curl -s https://api.github.com/repos/ggml-org/llama.cpp/releases/latest | grep -oP '"tag_name": "\K[^"]+')

# Get current docker image tag from Docker Hub
echo "Checking Docker Hub version..."
dockerhub_tag=$(curl -s "https://hub.docker.com/v2/repositories/edp1096/llama.cpp-spark/tags/?page_size=100" | grep -oP '"name":"\K[a-z][0-9]+-v[0-9]+' | head -n 1)

# Get local docker image tag
echo "Checking local image version..."
local_tag=$(docker images edp1096/llama.cpp-spark --format "{{.Tag}}" | grep -E '^[a-z][0-9]+' | head -n 1)

# Display versions
echo "===================="
echo "llama.cpp version: $llama_tag"
echo "Docker Hub version: ${dockerhub_tag:-none}"
echo "Local version: ${local_tag:-none}"
echo "===================="

# Compare Docker Hub and local, use the newer one
current_tag=""
if [ -n "$dockerhub_tag" ] && [ -n "$local_tag" ]; then
    result=$(compare_tags "$local_tag" "$dockerhub_tag")
    if [[ "$result" == "new_release" ]]; then
        current_tag="$local_tag"
        echo "Using local version (newer than Docker Hub)"
    else
        current_tag="$dockerhub_tag"
        echo "Using Docker Hub version"
    fi
elif [ -n "$dockerhub_tag" ]; then
    current_tag="$dockerhub_tag"
    echo "Using Docker Hub version"
elif [ -n "$local_tag" ]; then
    current_tag="$local_tag"
    echo "Using local version"
else
    current_tag="b0"
    echo "No current version found, using $current_tag"
fi

echo "Current version: $current_tag"

result=$(compare_tags "$llama_tag" "$current_tag")
if [[ "$result" == "new_release" ]]; then
    echo "Build required: $current_tag -> $llama_tag-v1"
    exit 0
elif [[ "$result" == "rollback" ]]; then
    echo "Rollback detected: $current_tag -> $llama_tag-v1"
    exit 0
else
    echo "Already up to date: $current_tag"
    exit 1
fi
