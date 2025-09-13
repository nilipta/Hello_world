#!/bin/bash

RAMDISK_PATH="/mnt/ramdisk"

# Get total RAM in KB and calculate 80%
TOTAL_RAM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
RAMDISK_SIZE_KB=$((TOTAL_RAM_KB * 80 / 100))
RAMDISK_SIZE_MB=$((RAMDISK_SIZE_KB / 1024))

# Get current user UID and GID
USER_UID=$(id -u)
USER_GID=$(id -g)

echo "Choose an action:"
echo "  [m] Mount RAM disk (${RAMDISK_SIZE_MB}MB)"
echo "  [u] Unmount RAM disk"
echo "  [n] Mount and scaffold Next.js app"
echo "  [q] Quit"
read -p "Enter your choice [m/u/n/q]: " choice

case "$choice" in
    [mM])
        if mountpoint -q "$RAMDISK_PATH"; then
            echo "RAM disk already mounted at $RAMDISK_PATH"
        else
            sudo mkdir -p "$RAMDISK_PATH"
            sudo mount -t tmpfs -o size=${RAMDISK_SIZE_KB}k,uid=$USER_UID,gid=$USER_GID tmpfs "$RAMDISK_PATH"
            echo "Mounted RAM disk at $RAMDISK_PATH with size ${RAMDISK_SIZE_MB}MB"
        fi
        ;;
    [uU])
        if mountpoint -q "$RAMDISK_PATH"; then
            sudo umount "$RAMDISK_PATH"
            echo "Unmounted RAM disk from $RAMDISK_PATH"
        else
            echo "No RAM disk mounted at $RAMDISK_PATH"
        fi
        ;;
    [nN])
        if ! mountpoint -q "$RAMDISK_PATH"; then
            echo "Mounting RAM disk first..."
            sudo mkdir -p "$RAMDISK_PATH"
            sudo mount -t tmpfs -o size=${RAMDISK_SIZE_KB}k,uid=$USER_UID,gid=$USER_GID tmpfs "$RAMDISK_PATH"
            echo "Mounted RAM disk at $RAMDISK_PATH with size ${RAMDISK_SIZE_MB}MB"
        fi

        echo "Setting up Next.js app in RAM disk..."
        export npm_config_cache=/tmp/.npm
        export NODE_OPTIONS="--no-warnings"
        cd "$RAMDISK_PATH"
        npx create-next-app@latest myapp --eslint --no-ts --no-tailwind
        ;;
    [qQ])
        echo "Exiting without changes."
        ;;
    *)
        echo "Invalid choice. Please enter m, u, n, or q."
        ;;
esac
