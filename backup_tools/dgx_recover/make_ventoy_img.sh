#!/bin/bash
# DGX Spark Recovery IMG - Ventoy Compatible
# Creates a FAT32 disk image (.img) instead of ISO
# Reason: kernel 6.14.0-1013-nvidia has vfat built-in but NOT iso9660
#
# Requires: Ventoy USB formatted as NTFS (ntfs-3g available in initrd)
#
# Patches:
#   init_mount - mounts NTFS USB via ntfs-3g, finds .img, loop-mounts FAT32 partition
#   init       - detects Ventoy (VTOYEFI partition) and forces scripted recovery mode
#
# Run in WSL: bash make_ventoy_img.sh
set -e

# === Configuration ===
# BASE="/mnt/d/dev/asus_ascent_gx10_dgx_spark"
BASE="$(cd "." && pwd)" # Current directory as base
RECOVERY_TAR="$BASE/dgx-spark-recovery-image-1.120.36.tar.gz"
USB_PATH="$BASE/dgx_usb"
OUTPUT_IMG="$BASE/dgx_img/dgx_spark_recovery_ventoy.img"
WORK_DIR="$BASE/dgx_ventoy_work"
PART_LABEL="BOOTME"

echo "=== DGX Spark Ventoy-Compatible IMG Builder ==="
echo "=== FAT32 format (kernel has vfat, NOT iso9660) ==="
echo ""

# === [1/6] Check dependencies ===
echo "[1/6] Checking dependencies..."
MISSING=""
for cmd in mcopy mmd mkfs.fat zstd cpio python3; do
    if ! command -v "$cmd" &>/dev/null; then
        MISSING="$MISSING $cmd"
    fi
done
if [ -n "$MISSING" ]; then
    echo "  Installing missing packages..."
    sudo apt update
    sudo apt install -y mtools dosfstools zstd
fi
echo "  OK"

# === Extract from tar.gz if specified ===
if [ -n "$RECOVERY_TAR" ]; then
    if [ ! -f "$RECOVERY_TAR" ]; then
        echo "ERROR: Recovery tar.gz not found: $RECOVERY_TAR"
        exit 1
    fi
    echo "[*] Extracting USB contents from $(basename "$RECOVERY_TAR")..."
    USB_PATH="$WORK_DIR/usbimg.customer/usb"
    mkdir -p "$WORK_DIR"
    tar xzf "$RECOVERY_TAR" -C "$WORK_DIR" usbimg.customer/usb/
    echo "  Extracted to $USB_PATH"
fi

# === [2/6] Verify USB backup ===
echo "[2/6] Verifying USB backup at $USB_PATH ..."
for f in vmlinuz initrd fastos-release.txt fastos.partaa fastos.partab; do
    if [ ! -f "$USB_PATH/$f" ]; then
        echo "ERROR: $f not found in $USB_PATH"
        exit 1
    fi
done
if [ ! -d "$USB_PATH/efi/boot" ]; then
    echo "ERROR: efi/boot not found in $USB_PATH"
    exit 1
fi
echo "  OK"

# === [3/6] Extract and patch initrd ===
echo "[3/6] Patching initrd for Ventoy compatibility..."
rm -rf "$WORK_DIR/initrd_extract" "$WORK_DIR/initrd2.zst" "$WORK_DIR/initrd2.cpio" "$WORK_DIR/initrd2_new.zst" "$WORK_DIR/initrd_patched" "$WORK_DIR/part.img"
mkdir -p "$WORK_DIR/initrd_extract"

# Auto-detect initrd structure
echo "  Detecting initrd offsets..."
TRAILER_OFF=$(grep -boa 'TRAILER!!!' "$USB_PATH/initrd" | tail -1 | cut -d: -f1)
if [ -z "$TRAILER_OFF" ]; then
    echo "ERROR: Could not find TRAILER!!! marker in initrd"
    exit 1
fi

ZSTD_REL=$(tail -c +$((TRAILER_OFF + 1)) "$USB_PATH/initrd" | grep -boa $'\x28\xb5\x2f\xfd' | head -1 | cut -d: -f1)
if [ -z "$ZSTD_REL" ]; then
    echo "ERROR: Could not find zstd compressed initramfs"
    exit 1
fi
INITRD_OFFSET=$((TRAILER_OFF + ZSTD_REL))
echo "  zstd initramfs at offset $INITRD_OFFSET"

# Extract second cpio
tail -c +$((INITRD_OFFSET + 1)) "$USB_PATH/initrd" > "$WORK_DIR/initrd2.zst"
zstd -d "$WORK_DIR/initrd2.zst" -o "$WORK_DIR/initrd2.cpio" 2>/dev/null
echo "  Decompressed: $(du -h "$WORK_DIR/initrd2.cpio" | cut -f1)"

cd "$WORK_DIR/initrd_extract"
cpio -idm < "$WORK_DIR/initrd2.cpio" 2>/dev/null
echo "  Extracted $(find . -type f | wc -l) files"

# --- Patch init_mount: add NTFS/ext4 Ventoy support (find .img -> loop-mount) ---
if [ ! -f init_mount ]; then
    echo "ERROR: init_mount not found in initramfs"
    exit 1
fi
cp init_mount init_mount.orig

cat > "$WORK_DIR/patch_init_mount.py" << 'PYEOF'
import sys, re

with open(sys.argv[1], 'r') as f:
    content = f.read()

ventoy_block = r'''
        # --- Ventoy: mount NTFS/ext4 USB, find .img, loop-mount FAT32 ---
        echo "  [Ventoy] blkid:"
        blkid 2>/dev/null || true
        mkdir -p /mnt/ventoy
        for usbdev in /dev/sd?1; do
            [ -e "$usbdev" ] || continue
            FSTYPE=$(blkid -s TYPE -o value "$usbdev" 2>/dev/null)
            echo "  [Ventoy] $usbdev: type=$FSTYPE"
            VMOUNTED=false
            if [ "$FSTYPE" = "ntfs" ]; then
                ntfs-3g "$usbdev" /mnt/ventoy -o ro 2>/dev/null && VMOUNTED=true
            elif [ "$FSTYPE" = "ext4" ] || [ "$FSTYPE" = "ext3" ] || [ "$FSTYPE" = "ext2" ]; then
                mount -o ro "$usbdev" /mnt/ventoy 2>/dev/null && VMOUNTED=true
            fi
            if [ "$VMOUNTED" = "true" ]; then
                    echo "  [Ventoy] Mounted $usbdev ($FSTYPE)"
                    for imgfile in /mnt/ventoy/*.img; do
                        [ -f "$imgfile" ] || continue
                        echo "  [Ventoy] Checking $imgfile"
                        PART_START=$(sfdisk -d "$imgfile" 2>/dev/null | grep "start=" | head -1 | sed 's/.*start= *\([0-9]*\).*/\1/')
                        if [ -n "$PART_START" ] && [ "$PART_START" -gt 0 ] 2>/dev/null; then
                            OFFSET=$((PART_START * 512))
                            echo "  [Ventoy] Partition at offset $OFFSET"
                            if losetup /dev/loop0 "$imgfile" -o "$OFFSET" 2>/dev/null; then
                                if mount /dev/loop0 /mnt/usb 2>/dev/null; then
                                    if [ -f /mnt/usb/fastos-release.txt ]; then
                                        echo "  [Ventoy] OK! Recovery files found"
                                        mounted=true
                                        break
                                    fi
                                    umount /mnt/usb 2>/dev/null
                                fi
                                losetup -d /dev/loop0 2>/dev/null
                            fi
                        fi
                    done
                    if [ "$mounted" != "true" ]; then
                        umount /mnt/ventoy 2>/dev/null
                    fi
            fi
            [ "$mounted" = "true" ] && break
        done
        if [ "$mounted" = "true" ]; then
            break
        fi
        # --- End Ventoy ---
'''

# Insert before "for device in /dev/sd"
match = re.search(r'^([ \t]*for device in /dev/sd)', content, re.MULTILINE)
if not match:
    print("ERROR: Could not find 'for device in /dev/sd' in init_mount", file=sys.stderr)
    sys.exit(1)

insert_pos = match.start()
patched = content[:insert_pos] + ventoy_block.rstrip() + '\n\n' + content[insert_pos:]

with open(sys.argv[1], 'w') as f:
    f.write(patched)

print("  Patched init_mount: NTFS Ventoy support added")
PYEOF

python3 "$WORK_DIR/patch_init_mount.py" "$WORK_DIR/initrd_extract/init_mount"

# --- Patch init: detect Ventoy and force noui (scripted recovery) ---
if [ ! -f init ]; then
    echo "ERROR: init not found in initramfs"
    exit 1
fi
cp init init.orig

cat > "$WORK_DIR/patch_init.py" << 'PYEOF'
import sys, re

with open(sys.argv[1], 'r') as f:
    content = f.read()

ventoy_detect = '''
# --- Ventoy detection: VTOYEFI partition = Ventoy USB boot ---
sleep 3
if blkid 2>/dev/null | grep -q 'VTOYEFI'; then
    echo "    - Ventoy boot detected (VTOYEFI found), using scripted recovery"
    noui=y
fi
# --- End Ventoy detection ---

'''

# Insert before 'if [ -n "$noui" ]'
match = re.search(r'^(if \[ -n "\$noui" \])', content, re.MULTILINE)
if not match:
    print("ERROR: Could not find noui check in init", file=sys.stderr)
    sys.exit(1)

pos = match.start()
patched = content[:pos] + ventoy_detect + content[pos:]

with open(sys.argv[1], 'w') as f:
    f.write(patched)

print("  Patched init: Ventoy detection added")
PYEOF

python3 "$WORK_DIR/patch_init.py" "$WORK_DIR/initrd_extract/init"

# Show diffs
echo ""
echo "  --- init_mount diff ---"
diff "$WORK_DIR/initrd_extract/init_mount.orig" "$WORK_DIR/initrd_extract/init_mount" || true
echo "  --- init diff ---"
diff "$WORK_DIR/initrd_extract/init.orig" "$WORK_DIR/initrd_extract/init" || true
echo ""

# === [4/6] Rebuild initrd ===
echo "[4/6] Rebuilding initrd..."
cd "$WORK_DIR/initrd_extract"
find . | cpio -o -H newc 2>/dev/null | zstd > "$WORK_DIR/initrd2_new.zst" 2>/dev/null

head -c "$INITRD_OFFSET" "$USB_PATH/initrd" > "$WORK_DIR/initrd_patched"
cat "$WORK_DIR/initrd2_new.zst" >> "$WORK_DIR/initrd_patched"
echo "  Original: $(du -h "$USB_PATH/initrd" | cut -f1)"
echo "  Patched:  $(du -h "$WORK_DIR/initrd_patched" | cut -f1)"

# === [5/6] Create FAT32 partition image ===
echo "[5/6] Creating FAT32 partition image..."

# Calculate required size
TOTAL_DATA=0
for f in "$USB_PATH"/vmlinuz "$USB_PATH"/fastos-release.txt "$USB_PATH"/fastos.partaa \
         "$USB_PATH"/fastos.partab "$USB_PATH"/efi.tar.xz "$WORK_DIR"/initrd_patched; do
    if [ -f "$f" ]; then
        FSIZE=$(stat -c%s "$f")
        TOTAL_DATA=$((TOTAL_DATA + FSIZE))
    fi
done
# Add efi, boot/grub, fw directory sizes
for d in "$USB_PATH/efi" "$USB_PATH/boot" "$USB_PATH/fw"; do
    if [ -d "$d" ]; then
        DSIZE=$(du -sb "$d" | awk '{print $1}')
        TOTAL_DATA=$((TOTAL_DATA + DSIZE))
    fi
done

# Add 300MB padding for FAT32 metadata + alignment
PART_SIZE_MB=$(( (TOTAL_DATA + 300*1024*1024) / 1024 / 1024 ))
echo "  Total data: $((TOTAL_DATA / 1024 / 1024))MB"
echo "  Partition size: ${PART_SIZE_MB}MB"

PART_IMG="$WORK_DIR/part.img"
dd if=/dev/zero of="$PART_IMG" bs=1M count=$PART_SIZE_MB status=none
mkfs.fat -F 32 -n "$PART_LABEL" -s 8 "$PART_IMG" >/dev/null
echo "  FAT32 formatted"

# Copy files to partition using mtools
echo "  Copying files to FAT32 partition..."

# Create directory structure
mmd -i "$PART_IMG" ::efi ::efi/boot ::boot ::boot/grub ::fw 2>/dev/null || true

# Copy EFI boot files
for f in "$USB_PATH"/efi/boot/*; do
    [ -f "$f" ] && mcopy -i "$PART_IMG" "$f" "::efi/boot/" 2>/dev/null
done
echo "    efi/boot OK"

# Copy GRUB files (including arm64-efi modules and fonts)
if [ -d "$USB_PATH/boot/grub" ]; then
    # Create subdirectories first
    find "$USB_PATH/boot/grub" -type d | sort | while read -r dir; do
        REL="${dir#$USB_PATH/boot/grub}"
        if [ -n "$REL" ]; then
            mmd -i "$PART_IMG" "::boot/grub${REL}" 2>/dev/null || true
        fi
    done
    # Copy files
    find "$USB_PATH/boot/grub" -type f | while read -r file; do
        REL="${file#$USB_PATH/}"
        mcopy -i "$PART_IMG" "$file" "::${REL}" 2>/dev/null || true
    done
    echo "    boot/grub OK"
fi

# Copy firmware files
if [ -d "$USB_PATH/fw" ]; then
    find "$USB_PATH/fw" -type d | sort | while read -r dir; do
        REL="${dir#$USB_PATH}"
        if [ -n "$REL" ] && [ "$REL" != "/fw" ]; then
            mmd -i "$PART_IMG" "::${REL}" 2>/dev/null || true
        fi
    done
    find "$USB_PATH/fw" -type f | while read -r file; do
        REL="${file#$USB_PATH/}"
        mcopy -i "$PART_IMG" "$file" "::${REL}" 2>/dev/null || true
    done
    echo "    fw OK"
fi

# Copy individual files
mcopy -i "$PART_IMG" "$WORK_DIR/initrd_patched" "::initrd"
echo "    initrd OK"
mcopy -i "$PART_IMG" "$USB_PATH/vmlinuz" "::vmlinuz"
echo "    vmlinuz OK"
mcopy -i "$PART_IMG" "$USB_PATH/fastos-release.txt" "::fastos-release.txt"
echo "    fastos-release.txt OK"
mcopy -i "$PART_IMG" "$USB_PATH/efi.tar.xz" "::efi.tar.xz"
echo "    efi.tar.xz OK"
[ -f "$USB_PATH/sgdisk.txt.example" ] && mcopy -i "$PART_IMG" "$USB_PATH/sgdisk.txt.example" "::sgdisk.txt.example"

# Copy large partition files
echo "    fastos.partaa ($(du -h "$USB_PATH/fastos.partaa" | cut -f1))..."
mcopy -i "$PART_IMG" "$USB_PATH/fastos.partaa" "::fastos.partaa"
echo "    fastos.partaa OK"
echo "    fastos.partab ($(du -h "$USB_PATH/fastos.partab" | cut -f1))..."
mcopy -i "$PART_IMG" "$USB_PATH/fastos.partab" "::fastos.partab"
echo "    fastos.partab OK"

echo "  Partition image: $(du -h "$PART_IMG" | cut -f1)"

# === [6/6] Assemble disk image with MBR ===
mkdir -p "$(dirname "$OUTPUT_IMG")"
echo "[6/6] Assembling disk image with MBR partition table..."

python3 - "$PART_IMG" "$OUTPUT_IMG" << 'PYEOF'
import sys, struct, os

part_file = sys.argv[1]
out_file = sys.argv[2]

part_size = os.path.getsize(part_file)
part_sectors = part_size // 512
part_start = 2048  # Standard 1MB alignment

# Build MBR (512 bytes)
mbr = bytearray(512)

# Partition entry 1 at offset 446 (16 bytes)
entry = bytearray(16)
entry[0] = 0x80           # Boot indicator (bootable)
entry[1:4] = b'\xfe\xff\xff'  # CHS start (dummy, LBA used)
entry[4] = 0x0C           # Partition type: FAT32 LBA
entry[5:8] = b'\xfe\xff\xff'  # CHS end (dummy, LBA used)
struct.pack_into('<I', entry, 8, part_start)      # LBA start
struct.pack_into('<I', entry, 12, part_sectors)    # LBA size
mbr[446:462] = entry

# Boot signature
mbr[510] = 0x55
mbr[511] = 0xAA

# Write output
header_size = part_start * 512  # 1MB
total_size = header_size + part_size

with open(out_file, 'wb') as out:
    # Write MBR + padding to 1MB
    out.write(bytes(mbr))
    out.write(b'\x00' * (header_size - 512))

    # Copy partition data in chunks
    copied = 0
    with open(part_file, 'rb') as part:
        while True:
            chunk = part.read(4 * 1024 * 1024)  # 4MB chunks
            if not chunk:
                break
            out.write(chunk)
            copied += len(chunk)
            pct = copied * 100 // part_size
            print(f"\r  Writing: {pct}%  ({copied // (1024*1024)}MB / {part_size // (1024*1024)}MB)", end='', flush=True)

print()
total_gb = total_size / (1024 * 1024 * 1024)
print(f"  Disk image: {total_gb:.2f} GB (MBR + FAT32 partition at sector {part_start})")
PYEOF

echo ""
echo "=== Done ==="
ls -lh "$OUTPUT_IMG"
echo ""
echo "IMG saved to: $OUTPUT_IMG"
echo ""
echo "Usage:"
echo "  1. Install Ventoy on USB with NTFS or ext4 filesystem (NOT exFAT)"
echo "  2. Copy the .img file to the Ventoy USB drive"
echo "  3. Boot from Ventoy and select the .img file"
echo ""
echo "How it works:"
echo "  Ventoy boots kernel+initrd from .img -> init detects VTOYEFI ->"
echo "  init_mount uses ntfs-3g to mount USB -> finds .img -> loop-mounts"
echo "  FAT32 partition inside .img -> recovery proceeds"

# Cleanup
echo ""
echo "Cleaning up work directory..."
rm -rf "$WORK_DIR"
echo "Complete."
