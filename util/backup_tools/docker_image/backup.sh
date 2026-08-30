#!/bin/bash
# backup.sh - Selective or full Docker image backup

BACKUP_DIR="/mnt/ssd_t5/docker_backup"

if [ "$EUID" -ne 0 ]; then
    echo "Error: This script must be run as root or with sudo"
    exit 1
fi

if [ ! -d "/mnt/ssd_t5" ]; then
    echo "Error: /mnt/ssd_t5 not found. External SSD may not be mounted"
    exit 1
fi

if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running"
    exit 1
fi

mkdir -p "$BACKUP_DIR"

show_images() {
    echo "Available Docker images:"
    docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}" | head -1
    docker images --format "{{.Repository}}:{{.Tag}}\t{{.Size}}" | grep -v "<none>" | nl
    echo ""
}

backup_image() {
    local image="$1"
    local safe_name=$(echo "$image" | tr '/:' '_')
    local backup_file="$BACKUP_DIR/${safe_name}.tar"
    
    echo "Backing up: $image"
    echo "  File: ${safe_name}.tar"
    
    docker save -o "$backup_file" "$image"
    
    if [ $? -eq 0 ]; then
        echo "  Success"
        du -sh "$backup_file"
    else
        echo "  Error: Failed to backup $image"
        return 1
    fi
    echo ""
}

if [ -z "$1" ]; then
    show_images
    
    read -p "Enter image number(s) to backup (comma-separated, or 'all' for all images): " selection
    
    images=$(docker images --format "{{.Repository}}:{{.Tag}}" | grep -v "<none>")
    
    if [ "$selection" = "all" ]; then
        echo ""
        echo "Backing up all images..."
        echo ""
        
        total=$(echo "$images" | wc -l)
        current=0
        
        for image in $images; do
            current=$((current + 1))
            echo "[$current/$total]"
            backup_image "$image"
        done
    else
        echo ""
        IFS=',' read -ra NUMS <<< "$selection"
        
        for num in "${NUMS[@]}"; do
            num=$(echo "$num" | xargs)
            image=$(echo "$images" | sed -n "${num}p")
            
            if [ -n "$image" ]; then
                backup_image "$image"
            else
                echo "Invalid selection: $num"
            fi
        done
    fi
else
    if [ "$1" = "all" ]; then
        images=$(docker images --format "{{.Repository}}:{{.Tag}}" | grep -v "<none>")
        total=$(echo "$images" | wc -l)
        current=0
        
        echo "Backing up all images..."
        echo ""
        
        for image in $images; do
            current=$((current + 1))
            echo "[$current/$total]"
            backup_image "$image"
        done
    else
        backup_image "$1"
    fi
fi

echo "Backup completed!"
echo "Backup directory size:"
du -sh "$BACKUP_DIR"
