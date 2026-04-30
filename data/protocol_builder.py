"""
protocol_builder.py
===================
One-time utility to patch Colab-style protocol .txt files so they point to
the local rclone mount path instead of /content/drive/MyDrive/...

Run ONCE after mounting Drive:
    python data/protocol_builder.py

After this the protocol files are frozen — do NOT re-run unless you mount
Drive at a different letter.
"""

import os
import sys

DRIVE_ROOT   = "X:\\"

COLAB_PREFIX = "/content/drive/MyDrive"

PROTOCOL_DIR = os.path.join(DRIVE_ROOT, "aasist_dataset_v2", "protocols")
SPLITS = ["train", "dev", "eval"]


def patch_protocol(split: str, dry_run: bool = False) -> int:
    """Replace Colab paths with Windows rclone paths. Returns number of lines changed."""
    path = os.path.join(PROTOCOL_DIR, f"custom_{split}.txt")
    if not os.path.isfile(path):
        print(f"  [!!] Not found: {path}")
        return 0

    with open(path, encoding="utf-8") as f:
        content = f.read()

    updated = content.replace(COLAB_PREFIX, DRIVE_ROOT)
    # Normalise any remaining forward slashes after the drive root
    updated = updated.replace("/", os.sep)

    changed = content != updated
    n_changes = sum(
        1 for a, b in zip(content.splitlines(), updated.splitlines()) if a != b
    )

    if dry_run:
        print(f"  [DRY-RUN] {split}: {n_changes} lines would be updated")
        return n_changes

    if changed:
        with open(path, "w", encoding="utf-8") as f:
            f.write(updated)
        print(f"  [OK] Updated {split}: {n_changes} lines patched -> {path}")
    else:
        print(f"  [OK] {split}: already uses Windows paths, no changes needed")

    return n_changes


def main():
    dry = "--dry-run" in sys.argv
    print(f"Protocol path-patcher  [dry_run={dry}]")
    print(f"Drive root : {DRIVE_ROOT}")
    print(f"Protocol dir: {PROTOCOL_DIR}\n")

    if not os.path.isdir(PROTOCOL_DIR):
        print(f"ERROR: Protocol dir not found. Is rclone mounted at X:\\?")
        sys.exit(1)

    total = 0
    for split in SPLITS:
        total += patch_protocol(split, dry_run=dry)

    print(f"\nDone. {total} total lines patched.")
    if total > 0 and not dry:
        print("Protocol files are now frozen — do not re-run this script.")


if __name__ == "__main__":
    main()
