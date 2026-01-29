"""enhance_cli.py

Tool nhỏ để test nhanh việc "photoshop" crop biển số ngoài UI.

Ví dụ:
  python enhance_cli.py --in snapshots/2026-01-28/PEND_xxx.jpg --out out.jpg --mode cpu+torch

Modes:
  - cpu        : dùng enhance_plate_bgr (OpenCV)
  - torch      : dùng TorchPlateEnhancer (GPU nếu có)
  - cpu+torch  : CPU trước, rồi torch

Tip:
  - Nếu bạn chạy được torch cuda, mode cpu+torch thường cho ảnh dễ OCR hơn.
"""

from __future__ import annotations

import argparse
import os
import cv2

from enhance_worker import enhance_plate_bgr

try:
    from gpu_enhance import TorchPlateEnhancer
except Exception:
    TorchPlateEnhancer = None


def _lap_var(bgr):
    try:
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(g, cv2.CV_64F).var())
    except Exception:
        return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="input image path")
    ap.add_argument("--out", dest="out", required=True, help="output image path")
    ap.add_argument("--mode", default="cpu+torch", choices=["cpu", "torch", "cpu+torch"], help="enhance mode")
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--amount", type=float, default=1.0)
    ap.add_argument("--contrast", type=float, default=1.15)
    args = ap.parse_args()

    img = cv2.imread(args.inp)
    if img is None:
        raise SystemExit(f"Cannot read: {args.inp}")

    before = _lap_var(img)

    out = img
    if args.mode == "cpu":
        out = enhance_plate_bgr(out)
    elif args.mode == "torch":
        if TorchPlateEnhancer is None:
            raise SystemExit("TorchPlateEnhancer not available. Install torch + CUDA first.")
        enh = TorchPlateEnhancer(scale=args.scale, amount=args.amount, contrast=args.contrast)
        out = enh.enhance(out)
    else:  # cpu+torch
        out = enhance_plate_bgr(out)
        if TorchPlateEnhancer is None:
            raise SystemExit("TorchPlateEnhancer not available. Install torch + CUDA first.")
        enh = TorchPlateEnhancer(scale=args.scale, amount=args.amount, contrast=args.contrast)
        out = enh.enhance(out)

    after = _lap_var(out)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    cv2.imwrite(args.out, out)

    print(f"Saved: {args.out}")
    print(f"Sharpness: before={before:.1f} after={after:.1f}")


if __name__ == "__main__":
    main()
