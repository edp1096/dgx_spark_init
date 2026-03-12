#!/bin/bash
# DGX Spark Recovery ISO - Ventoy Compatible
# Patches init_mount to scan Ventoy devices (dm-*, loop*, sr*) then rebuilds ISO
# Run in WSL: bash make_ventoy_iso.sh
set -e

# === Configuration ===
BASE="/mnt/d/dev/asus_ascent_gx10_dgx_spark"
USB_PATH="$BASE/dgx_usb"
OUTPUT_ISO="$BASE/dgx_iso/dgx_spark_recovery_ventoy.iso"
WORK_DIR="$BASE/dgx_ventoy_work"
ISO_LABEL="BOOTME"

echo "=== DGX Spark Ventoy-Compatible ISO Builder ==="
echo ""

# === Check dependencies ===
echo "[1/6] Checking dependencies..."
MISSING=""
for cmd in xorriso mmd mcopy mkfs.fat zstd cpio; do
    if ! command -v "$cmd" &>/dev/null; then
        MISSING="$MISSING $cmd"
    fi
done
if [ -n "$MISSING" ]; then
    echo "  Installing missing packages..."
    sudo apt update
    sudo apt install -y xorriso mtools dosfstools zstd
fi
echo "  OK"

# === Verify USB backup ===
echo "[2/6] Verifying USB backup at $USB_PATH ..."
for f in vmlinuz initrd fastos-release.txt; do
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

# === Extract and patch initrd ===
echo "[3/6] Patching initrd for Ventoy compatibility..."
rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR/initrd_extract"

# Auto-detect initrd structure: find TRAILER!!! then zstd magic after it
echo "  Auto-detecting initrd offsets..."
TRAILER_OFF=$(grep -boa 'TRAILER!!!' "$USB_PATH/initrd" | tail -1 | cut -d: -f1)
if [ -z "$TRAILER_OFF" ]; then
    echo "ERROR: Could not find TRAILER!!! marker in initrd"
    exit 1
fi
echo "  TRAILER!!! found at offset $TRAILER_OFF"

# Search for zstd magic (0x28B52FFD) after TRAILER
ZSTD_REL=$(tail -c +$((TRAILER_OFF + 1)) "$USB_PATH/initrd" | grep -boa $'\x28\xb5\x2f\xfd' | head -1 | cut -d: -f1)
if [ -z "$ZSTD_REL" ]; then
    echo "ERROR: Could not find zstd compressed initramfs after TRAILER"
    exit 1
fi
INITRD_OFFSET=$((TRAILER_OFF + ZSTD_REL))
echo "  zstd initramfs detected at offset $INITRD_OFFSET"

# Extract second cpio (zstd compressed initramfs)
tail -c +$((INITRD_OFFSET + 1)) "$USB_PATH/initrd" > "$WORK_DIR/initrd2.zst"
echo "  Compressed size: $(du -h "$WORK_DIR/initrd2.zst" | cut -f1)"

zstd -d "$WORK_DIR/initrd2.zst" -o "$WORK_DIR/initrd2.cpio" 2>/dev/null
echo "  Decompressed size: $(du -h "$WORK_DIR/initrd2.cpio" | cut -f1)"

# Extract cpio contents
cd "$WORK_DIR/initrd_extract"
cpio -idm < "$WORK_DIR/initrd2.cpio" 2>/dev/null
echo "  Extracted $(find . -type f | wc -l) files"

# Verify init_mount exists
if [ ! -f init_mount ]; then
    echo "ERROR: init_mount not found in initramfs"
    exit 1
fi

# Backup original
cp init_mount init_mount.orig

# Patch init_mount: add Ventoy device scanning before the sd?1 loop
# The patch inserts a block that tries dm-*, loop*, sr* devices with iso9660/auto mount
cat > "$WORK_DIR/patch_init_mount.py" << 'PYEOF'
import sys

with open(sys.argv[1], 'r') as f:
    content = f.read()

# The Ventoy block to insert before the sd?1 scanning
ventoy_block = '''
        # --- Ventoy compatibility: try device-mapper, loop, CD-ROM devices ---
        for device in /dev/dm-* /dev/loop* /dev/sr*; do
            [ -e "$device" ] || continue
            for fstype in iso9660 udf auto; do
                if mount -t "$fstype" -o ro "$device" /mnt/usb 2>/dev/null; then
                    echo "Mounted $device ($fstype), checking for fastos-release.txt..."
                    if [ -f /mnt/usb/fastos-release.txt ]; then
                        echo "Found fastos-release.txt on $device (Ventoy/ISO mode)"
                        mounted=true
                        break 2
                    else
                        umount /mnt/usb 2>/dev/null
                    fi
                fi
            done
        done
        if [ "$mounted" = "true" ]; then
            break
        fi
        # --- End Ventoy compatibility ---
'''

# Find the insertion point: the line with "for device in /dev/sd"
# This is more stable than matching a comment that NVIDIA might change
import re
match = re.search(r'^([ \t]*for device in /dev/sd)', content, re.MULTILINE)
if not match:
    print("ERROR: Could not find 'for device in /dev/sd' in init_mount", file=sys.stderr)
    sys.exit(1)

insert_pos = match.start()
patched = content[:insert_pos] + ventoy_block.rstrip() + '\n\n' + content[insert_pos:]

with open(sys.argv[1], 'w') as f:
    f.write(patched)

print("  Patched init_mount successfully")
PYEOF

python3 "$WORK_DIR/patch_init_mount.py" "$WORK_DIR/initrd_extract/init_mount"

# Show diff
echo "  --- Diff ---"
diff "$WORK_DIR/initrd_extract/init_mount.orig" "$WORK_DIR/initrd_extract/init_mount" || true
echo "  --- End Diff ---"

# === Rebuild initrd ===
echo "[4/6] Rebuilding initrd..."

# Rebuild second cpio (SVR4/newc format, zstd compressed)
cd "$WORK_DIR/initrd_extract"
find . | cpio -o -H newc 2>/dev/null | zstd > "$WORK_DIR/initrd2_new.zst" 2>/dev/null
echo "  New compressed initramfs: $(du -h "$WORK_DIR/initrd2_new.zst" | cut -f1)"

# Combine: first cpio (microcode/modules) + patched second cpio
head -c "$INITRD_OFFSET" "$USB_PATH/initrd" > "$WORK_DIR/initrd_patched"
cat "$WORK_DIR/initrd2_new.zst" >> "$WORK_DIR/initrd_patched"
echo "  Original initrd: $(du -h "$USB_PATH/initrd" | cut -f1)"
echo "  Patched initrd:  $(du -h "$WORK_DIR/initrd_patched" | cut -f1)"

# === Build EFI boot image ===
echo "[5/6] Creating EFI boot image..."
EFI_IMG="$WORK_DIR/efi_boot.img"
STAGE_DIR="$WORK_DIR/stage"
mkdir -p "$STAGE_DIR/boot"

# Calculate EFI image size
EFI_SIZE_KB=0
if [ -d "$USB_PATH/efi" ]; then
    EFI_SIZE_KB=$(du -sk "$USB_PATH/efi" | awk '{print $1}')
fi
if [ -d "$USB_PATH/boot/grub" ]; then
    GRUB_SIZE_KB=$(du -sk "$USB_PATH/boot/grub" | awk '{print $1}')
    EFI_SIZE_KB=$((EFI_SIZE_KB + GRUB_SIZE_KB))
fi
EFI_SIZE_KB=$(( (EFI_SIZE_KB * 3 / 2 + 1023) / 1024 * 1024 ))
[ "$EFI_SIZE_KB" -lt 4096 ] && EFI_SIZE_KB=4096

echo "  EFI image size: ${EFI_SIZE_KB}KB"
dd if=/dev/zero of="$EFI_IMG" bs=1K count="$EFI_SIZE_KB" 2>/dev/null
mkfs.fat -F 12 "$EFI_IMG" >/dev/null

# Copy EFI boot files
mmd -i "$EFI_IMG" ::EFI ::EFI/boot
for f in "$USB_PATH"/efi/boot/*; do
    [ -f "$f" ] && mcopy -i "$EFI_IMG" "$f" ::EFI/boot/
done

# Copy GRUB config
if [ -d "$USB_PATH/boot/grub" ]; then
    mmd -i "$EFI_IMG" ::boot ::boot/grub
    find "$USB_PATH/boot/grub" -type d | while read -r dir; do
        REL="${dir#$USB_PATH/boot/grub}"
        [ -n "$REL" ] && mmd -i "$EFI_IMG" "::boot/grub${REL}" 2>/dev/null || true
    done
    find "$USB_PATH/boot/grub" -type f | while read -r file; do
        REL="${file#$USB_PATH/}"
        mcopy -i "$EFI_IMG" "$file" "::${REL}" 2>/dev/null || true
    done
fi

cp "$EFI_IMG" "$STAGE_DIR/boot/efi_boot.img"

# === Stage files ===
echo "  Staging ISO contents..."
for item in "$USB_PATH"/*; do
    BASENAME=$(basename "$item")
    case "$BASENAME" in
        "System Volume Information") continue ;;
        "initrd")
            # Use patched initrd instead of original
            cp "$WORK_DIR/initrd_patched" "$STAGE_DIR/initrd"
            ;;
        "boot")
            cp -r "$item"/* "$STAGE_DIR/boot/" 2>/dev/null || true
            ;;
        *)
            cp -r "$item" "$STAGE_DIR/" 2>/dev/null || true
            ;;
    esac
done

echo "  Staging size: $(du -sh "$STAGE_DIR" | cut -f1)"

# === Generate ISO ===
echo "[6/6] Generating bootable ISO..."
xorriso -as mkisofs \
    -o "$OUTPUT_ISO" \
    -J -R \
    -V "$ISO_LABEL" \
    -eltorito-alt-boot \
    -e boot/efi_boot.img \
    -no-emul-boot \
    -isohybrid-gpt-basdat \
    "$STAGE_DIR"

echo ""
echo "=== Done ==="
ls -lh "$OUTPUT_ISO"
echo "ISO saved to: $OUTPUT_ISO"

# Cleanup
echo "Cleaning up work directory..."
rm -rf "$WORK_DIR"
echo "Complete."
