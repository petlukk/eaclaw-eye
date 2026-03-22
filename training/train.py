#!/usr/bin/env python3
"""
Train the eaclaw-eye CNN model on COCO detection crops.

Architecture matches crates/eye-core/src/vision/model.rs exactly:
  3-layer conv net: 1→8→16→32 channels, 3x3 kernels
  Each layer: Conv2d + ReLU + MaxPool2d(2)
  Output: GlobalAvgPool + FC(32→4) + Softmax
  Total: 6020 int8 parameters (~6 KB)

Classes:
  0 = nothing  (random background crops)
  1 = person
  2 = vehicle  (car, truck, bus, motorcycle, bicycle)
  3 = animal   (cat, dog, bird, horse, cow, sheep, elephant, bear, zebra, giraffe)

Usage:
  python train.py --coco-dir /path/to/coco --output model.bin
  python train.py --coco-dir /path/to/coco --output model.bin --epochs 30 --lr 0.01
"""

import argparse
import os
import random
import struct
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# COCO category IDs → eaclaw-eye class IDs
PERSON_IDS = {1}
VEHICLE_IDS = {2, 3, 4, 5, 7}        # bicycle, car, motorcycle, bus, truck
ANIMAL_IDS = {16, 17, 18, 19, 20, 21, 22, 23, 24, 25}  # bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe

TOTAL_PARAMS = 6020
INPUT_SIZE = 64
NUM_CLASSES = 4
CLASS_NAMES = ["nothing", "person", "vehicle", "animal"]


def coco_cat_to_class(cat_id: int) -> int:
    if cat_id in PERSON_IDS:
        return 1
    if cat_id in VEHICLE_IDS:
        return 2
    if cat_id in ANIMAL_IDS:
        return 3
    return -1  # skip


class EyeNet(nn.Module):
    """Matches the Rust inference engine exactly."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1, bias=True)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(32, NUM_CLASSES)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 64→32
        x = self.pool(F.relu(self.conv2(x)))  # 32→16
        x = self.pool(F.relu(self.conv3(x)))  # 16→8
        x = x.mean(dim=[2, 3])                # GAP: 8x8→1
        x = self.fc(x)
        return x

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


def export_int8(model: EyeNet, output_path: str, scale: float):
    """Export model weights as 6020-byte int8 binary matching Rust layout.

    Weight layout (contiguous):
      conv1_w: [8][1][3][3]   = 72 bytes
      conv1_b: [8]            = 8 bytes
      conv2_w: [16][8][3][3]  = 1152 bytes
      conv2_b: [16]           = 16 bytes
      conv3_w: [32][16][3][3] = 4608 bytes
      conv3_b: [32]           = 32 bytes
      fc_w:    [32][4]        = 128 bytes   (Rust: fc_weights[ch * NUM_CLASSES + c])
      fc_b:    [4]            = 4 bytes
    """
    buf = bytearray()

    def quantize(tensor: torch.Tensor) -> bytes:
        # Scale to [-127, 127] range and clamp
        scaled = (tensor.detach().float() * scale).round().clamp(-127, 127)
        return bytes(scaled.to(torch.int8).numpy().tobytes())

    # Conv layers: PyTorch layout is [c_out, c_in, kH, kW] — matches Rust
    buf += quantize(model.conv1.weight.data)  # [8, 1, 3, 3]
    buf += quantize(model.conv1.bias.data)    # [8]
    buf += quantize(model.conv2.weight.data)  # [16, 8, 3, 3]
    buf += quantize(model.conv2.bias.data)    # [16]
    buf += quantize(model.conv3.weight.data)  # [32, 16, 3, 3]
    buf += quantize(model.conv3.bias.data)    # [32]

    # FC layer: PyTorch is [out_features, in_features] = [4, 32]
    # Rust expects [in_features][out_features] = [32][4]
    # So we transpose
    fc_w = model.fc.weight.data.T.contiguous()  # [32, 4]
    buf += quantize(fc_w)
    buf += quantize(model.fc.bias.data)         # [4]

    assert len(buf) == TOTAL_PARAMS, f"expected {TOTAL_PARAMS} bytes, got {len(buf)}"

    with open(output_path, "wb") as f:
        f.write(buf)

    return buf


class CocoDetectionCrops(Dataset):
    """Loads COCO bounding box crops as 64x64 grayscale patches."""

    def __init__(self, coco_dir: str, split: str = "train", max_per_class: int = 5000):
        from pycocotools.coco import COCO

        ann_file = os.path.join(coco_dir, "annotations", f"instances_{split}2017.json")
        img_dir = os.path.join(coco_dir, f"{split}2017")

        if not os.path.exists(ann_file):
            print(f"Error: {ann_file} not found")
            print("Download COCO 2017:")
            print("  wget http://images.cocodataset.org/zips/train2017.zip")
            print("  wget http://images.cocodataset.org/zips/val2017.zip")
            print("  wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip")
            sys.exit(1)

        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # maps to [-1, 1] range matching Rust normalize
        ])

        # Collect crops: (image_id, bbox, class_id)
        self.samples = []
        class_counts = {0: 0, 1: 0, 2: 0, 3: 0}

        for ann in self.coco.loadAnns(self.coco.getAnnIds()):
            cls = coco_cat_to_class(ann["category_id"])
            if cls < 0:
                continue
            if class_counts[cls] >= max_per_class:
                continue
            # Skip tiny boxes
            _, _, bw, bh = ann["bbox"]
            if bw < 10 or bh < 10:
                continue
            self.samples.append((ann["image_id"], ann["bbox"], cls))
            class_counts[cls] += 1

        # Generate "nothing" crops — random regions from images with no relevant objects
        nothing_img_ids = []
        relevant_img_ids = set(s[0] for s in self.samples)
        all_img_ids = self.coco.getImgIds()
        random.shuffle(all_img_ids)

        for img_id in all_img_ids:
            if img_id not in relevant_img_ids:
                nothing_img_ids.append(img_id)
            if len(nothing_img_ids) >= max_per_class:
                break

        for img_id in nothing_img_ids:
            img_info = self.coco.loadImgs(img_id)[0]
            w, h = img_info["width"], img_info["height"]
            # Random crop from the image
            crop_size = min(w, h, max(64, min(w, h) // 2))
            cx = random.randint(0, max(0, w - crop_size))
            cy = random.randint(0, max(0, h - crop_size))
            self.samples.append((img_id, [cx, cy, crop_size, crop_size], 0))
            class_counts[0] += 1

        random.shuffle(self.samples)
        print(f"Dataset ({split}): {len(self.samples)} samples")
        for cls_id, name in enumerate(CLASS_NAMES):
            print(f"  {name}: {class_counts[cls_id]}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_id, bbox, cls = self.samples[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info["file_name"])

        img = Image.open(img_path).convert("RGB")
        x, y, w, h = [int(v) for v in bbox]
        # Clamp to image bounds
        iw, ih = img.size
        x = max(0, min(x, iw - 1))
        y = max(0, min(y, ih - 1))
        w = min(w, iw - x)
        h = min(h, ih - y)
        if w < 2 or h < 2:
            w, h = max(w, 2), max(h, 2)

        crop = img.crop((x, y, x + w, y + h))
        tensor = self.transform(crop)
        return tensor, cls


def compute_quant_scale(model: EyeNet) -> float:
    """Find scale factor that maps max abs weight to 127."""
    max_abs = 0.0
    for p in model.parameters():
        val = p.detach().abs().max().item()
        if val > max_abs:
            max_abs = val
    if max_abs == 0:
        return 1.0
    return 127.0 / max_abs


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = EyeNet().to(device)
    print(f"EyeNet: {model.param_count()} parameters (expect {TOTAL_PARAMS})")
    assert model.param_count() == TOTAL_PARAMS

    train_data = CocoDetectionCrops(args.coco_dir, "train", max_per_class=args.max_per_class)
    val_data = CocoDetectionCrops(args.coco_dir, "val", max_per_class=args.max_per_class // 5)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0

    for epoch in range(args.epochs):
        # Train
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_acc = 100.0 * correct / total
        avg_loss = total_loss / len(train_loader)

        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        class_correct = [0] * NUM_CLASSES
        class_total = [0] * NUM_CLASSES

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                for i in range(targets.size(0)):
                    t = targets[i].item()
                    class_total[t] += 1
                    if predicted[i].item() == t:
                        class_correct[t] += 1

        val_acc = 100.0 * val_correct / val_total
        scheduler.step()

        per_class = "  ".join(
            f"{CLASS_NAMES[c]}:{100.0 * class_correct[c] / max(1, class_total[c]):.0f}%"
            for c in range(NUM_CLASSES)
        )
        print(f"Epoch {epoch + 1:3d}/{args.epochs}  loss={avg_loss:.4f}  "
              f"train={train_acc:.1f}%  val={val_acc:.1f}%  [{per_class}]")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), args.output.replace(".bin", ".pt"))

    # Export best model as int8
    model.load_state_dict(torch.load(args.output.replace(".bin", ".pt"), weights_only=True))
    model.eval()

    scale = compute_quant_scale(model)
    print(f"\nQuantization scale: {scale:.2f} (max_abs_weight: {127.0 / scale:.4f})")

    export_int8(model, args.output, scale)
    print(f"Exported {TOTAL_PARAMS} bytes to {args.output}")
    print(f"Best validation accuracy: {best_acc:.1f}%")

    # Verify round-trip: load back and check
    with open(args.output, "rb") as f:
        data = f.read()
    assert len(data) == TOTAL_PARAMS
    print(f"Verified: {len(data)} bytes, ready for Model::from_bytes()")


def main():
    parser = argparse.ArgumentParser(description="Train eaclaw-eye CNN model")
    parser.add_argument("--coco-dir", required=True, help="Path to COCO dataset root (with annotations/ and train2017/)")
    parser.add_argument("--output", default="model.bin", help="Output binary weights (default: model.bin)")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs (default: 30)")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate (default: 0.01)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (default: 128)")
    parser.add_argument("--max-per-class", type=int, default=5000, help="Max samples per class (default: 5000)")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
