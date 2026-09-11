#!/usr/bin/env python3
"""Preprocess a JPEG image into a raw float32 NCHW binary input file
for the compiled Neutron TFLite models (input tensor: 'images', shape 1x3x640x640, float32).

Usage:
  python3 preprocess_image.py --image face.jpg --out face_scrfd.bin --mode meanstd
  python3 preprocess_image.py --image person.jpg --out person_yolo.bin --mode scale01
"""
import argparse
from PIL import Image
import numpy as np


def preprocess(path, out_path, mode, size=640):
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32)  # HWC, RGB

    if mode == "meanstd":
        # SCRFD preprocessing (matches insightface training config: mean=127.5, std=128.0)
        arr = (arr - 127.5) / 128.0
    elif mode == "scale01":
        # YOLOv8 preprocessing: scale to [0, 1]
        arr = arr / 255.0
    else:
        raise ValueError(f"unknown mode {mode}")

    chw = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
    nchw = np.expand_dims(chw, 0).astype(np.float32)
    nchw.tofile(out_path)
    print(f"{out_path}: shape={nchw.shape} dtype={nchw.dtype} bytes={nchw.nbytes}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["meanstd", "scale01"], required=True)
    ap.add_argument("--size", type=int, default=640)
    args = ap.parse_args()
    preprocess(args.image, args.out, args.mode, args.size)
