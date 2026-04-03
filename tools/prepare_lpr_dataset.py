"""
Prepare Thai License Plate dataset for PaddleOCR training.

Reads an .xlsx file with columns: filename, label
Crops the UPPER part of license plates from images.
Splits into train/val/test sets.
Outputs PaddleOCR-format label files (tab-separated).

Usage:
    python tools/prepare_lpr_dataset.py \
        --xlsx path/to/labels.xlsx \
        --image_dir path/to/original/images \
        --output_dir ./train_data/thai_lpr \
        --crop_upper  # optional: crop upper half of plate
        --train_ratio 0.8 \
        --val_ratio 0.1 \
        --test_ratio 0.1
"""

import argparse
import os
import random
import shutil

import cv2
import numpy as np

try:
    import openpyxl
except ImportError:
    print("Please install openpyxl: pip install openpyxl")
    exit(1)


def read_xlsx(xlsx_path):
    """Read xlsx file and return list of (filename, label) tuples."""
    wb = openpyxl.load_workbook(xlsx_path, read_only=True)
    ws = wb.active

    data = []
    header_skipped = False
    for row in ws.iter_rows(values_only=True):
        if not header_skipped:
            header_skipped = True
            continue
        filename, label = row[0], row[1]
        if filename and label:
            data.append((str(filename).strip(), str(label).strip()))

    wb.close()
    return data


def crop_upper_half(image):
    """Crop the upper portion of a license plate image (where '2กข 2556' is)."""
    h, w = image.shape[:2]
    # Upper ~55% of the plate typically contains the registration number
    upper = image[0:int(h * 0.55), :]
    return upper


def main():
    parser = argparse.ArgumentParser(description="Prepare Thai LPR dataset for PaddleOCR")
    parser.add_argument("--xlsx", required=True, help="Path to .xlsx label file")
    parser.add_argument("--image_dir", required=True, help="Directory containing original plate images")
    parser.add_argument("--output_dir", default="./train_data/thai_lpr", help="Output directory")
    parser.add_argument("--crop_upper", action="store_true", help="Crop upper half of plate images")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    assert abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    random.seed(args.seed)

    # Read labels
    data = read_xlsx(args.xlsx)
    print(f"Read {len(data)} entries from {args.xlsx}")

    # Shuffle and split
    random.shuffle(data)
    n = len(data)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)

    splits = {
        "train": data[:n_train],
        "val": data[n_train:n_train + n_val],
        "test": data[n_train + n_val:],
    }

    # Create output directories
    for split_name in splits:
        os.makedirs(os.path.join(args.output_dir, split_name), exist_ok=True)

    # Process each split
    for split_name, split_data in splits.items():
        label_lines = []
        skipped = 0

        for filename, label in split_data:
            src_path = os.path.join(args.image_dir, filename)
            if not os.path.exists(src_path):
                print(f"  WARNING: {src_path} not found, skipping")
                skipped += 1
                continue

            dst_filename = filename.replace(os.sep, "_")  # flatten any subdirs
            dst_path = os.path.join(args.output_dir, split_name, dst_filename)

            if args.crop_upper:
                img = cv2.imread(src_path)
                if img is None:
                    print(f"  WARNING: Cannot read {src_path}, skipping")
                    skipped += 1
                    continue
                cropped = crop_upper_half(img)
                cv2.imwrite(dst_path, cropped)
            else:
                shutil.copy2(src_path, dst_path)

            # PaddleOCR format: relative_path\tlabel
            rel_path = os.path.join("thai_lpr", split_name, dst_filename)
            label_lines.append(f"{rel_path}\t{label}")

        # Write label file
        label_file = os.path.join(args.output_dir, f"{split_name}_list.txt")
        with open(label_file, "w", encoding="utf-8") as f:
            f.write("\n".join(label_lines))

        print(f"  {split_name}: {len(label_lines)} images (skipped {skipped})")

    print(f"\nDone! Dataset saved to {args.output_dir}")
    print(f"Label files: train_list.txt, val_list.txt, test_list.txt")


if __name__ == "__main__":
    main()
