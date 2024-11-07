#!/bin/bash

# Set the size of the RAM disk
SIZE_MB=5
MOUNT_POINT="./ram"

# Detect the operating system
OS=$(uname)

# macOS RAM Disk creation
if [[ "$OS" == "Darwin" ]]; then
    echo "Creating RAM disk on macOS..."
    # Convert size to 512-byte sectors for macOS (5 MB * 2048 sectors per MB)
    SECTORS=$((SIZE_MB * 2048))
    # Create and mount the RAM disk
    DISK_ID=$(hdiutil attach -nomount ram://$SECTORS)
    diskutil erasevolume HFS+ "RAMDisk" $DISK_ID
    mkdir -p "$MOUNT_POINT"
    mount -t hfs $DISK_ID "$MOUNT_POINT"
    echo "5 MB RAM disk created and mounted at $MOUNT_POINT."
    echo "To unmount the RAM disk, use: diskutil eject $DISK_ID"

# Linux RAM Disk creation
elif [[ "$OS" == "Linux" ]]; then
    echo "Creating RAM disk on Linux..."
    # Convert size to KB for Linux (5 MB * 1024 KB per MB)
    SIZE_KB=$((SIZE_MB * 1024))
    # Create the RAM disk
    mkdir -p "$MOUNT_POINT"
    mount -t tmpfs -o size=${SIZE_KB}K tmpfs "$MOUNT_POINT"
    echo "5 MB RAM disk created and mounted at $MOUNT_POINT."
    echo "To unmount the RAM disk, use: umount $MOUNT_POINT"

else
    echo "Unsupported operating system: $OS"
    exit 1
fi
