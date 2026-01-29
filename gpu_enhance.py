"""gpu_enhance.py

Mục tiêu: tăng chất lượng crop biển số trước OCR theo kiểu "CCTV thương mại":
- Không block luồng realtime.
- Chỉ chạy ở EnhanceWorker (slow path) và chỉ khi cần.
- Tận dụng GPU (PyTorch) để upscale + unsharp mask + tăng tương phản.

Lưu ý quan trọng (thực tế):
- Không có thuật toán nào "cứu" 100% chi tiết đã mất do motion blur nặng.
- Nhưng upscale + sharpening + contrast đúng cách thường giúp OCR ổn định hơn.
- Nếu bạn muốn "AI" mạnh hơn (Real-ESRGAN / DeblurGAN), bạn có thể cắm thêm ở đây.

Không yêu cầu thêm thư viện ngoài torch + numpy.
"""

from __future__ import annotations

import threading
import numpy as np

try:
    import torch
    import torch.nn.functional as F
except Exception as e:  # pragma: no cover
    torch = None  # type: ignore
    F = None  # type: ignore


# Global lock: 2 camera = 2 EnhanceWorker threads.
# Serialize GPU enhance to avoid GPU thrash/OOM on laptop GPUs.
_GPU_LOCK = threading.Lock()


def _gaussian_kernel2d(k: int, sigma: float, device, dtype):
    # k should be odd
    ax = torch.arange(k, device=device, dtype=torch.float32) - (k - 1) / 2.0
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    kernel = kernel / torch.sum(kernel)
    kernel = kernel.to(device=device, dtype=dtype)
    return kernel


class TorchPlateEnhancer:
    """Lightweight GPU enhancer (torch) for plate crops.

    What it does:
      1) Upscale (bicubic)
      2) Gaussian blur
      3) Unsharp mask: img + amount*(img - blur)
      4) Simple contrast around mean

    This is intentionally lightweight & deterministic.

    Parameters:
      - scale: upscale factor (2.0 is usually enough)
      - amount: sharpening strength (0.8~1.2)
      - sigma: blur sigma for unsharp (0.8~1.2)
      - contrast: 1.0 = no change; 1.1~1.25 often helps
      - fp16: use float16 on CUDA for speed
    """

    def __init__(
        self,
        device: str | None = None,
        scale: float = 2.0,
        amount: float = 1.0,
        sigma: float = 1.0,
        contrast: float = 1.15,
        fp16: bool = True,
    ):
        if torch is None:
            raise RuntimeError("Torch is not available")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.scale = float(scale)
        self.amount = float(amount)
        self.sigma = float(sigma)
        self.contrast = float(contrast)
        self.fp16 = bool(fp16) and (self.device.type == "cuda")

        # Precompute gaussian kernel (odd k)
        k = 7
        dtype = torch.float16 if self.fp16 else torch.float32
        kernel2d = _gaussian_kernel2d(k=k, sigma=max(0.1, self.sigma), device=self.device, dtype=dtype)
        # Shape for grouped conv2d: (C,1,k,k)
        self._kernel = kernel2d.view(1, 1, k, k).repeat(3, 1, 1, 1).contiguous()
        self._pad = k // 2

        # Warmup (optional) to reduce first-call latency
        if self.device.type == "cuda":
            try:
                dummy = torch.zeros((1, 3, 64, 160), device=self.device, dtype=dtype)
                _ = F.interpolate(dummy, scale_factor=2.0, mode="bicubic", align_corners=False)
                _ = F.conv2d(dummy, self._kernel, padding=self._pad, groups=3)
            except Exception:
                pass

    def enhance(self, bgr: np.ndarray) -> np.ndarray:
        """Input/Output: BGR uint8 image (numpy)."""
        if bgr is None or bgr.size == 0:
            return bgr
        if torch is None:
            return bgr

        # Ensure HWC uint8
        if bgr.dtype != np.uint8:
            bgr = np.clip(bgr, 0, 255).astype(np.uint8)

        # Avoid tiny crops
        h, w = bgr.shape[:2]
        if h < 10 or w < 10:
            return bgr

        # NOTE: we keep BGR channel order; operations are channel-wise, so order doesn't matter.
        with _GPU_LOCK:
            dtype = torch.float16 if self.fp16 else torch.float32

            t = torch.from_numpy(np.ascontiguousarray(bgr)).to(self.device)
            t = t.to(dtype=dtype)
            t = t.permute(2, 0, 1).unsqueeze(0)  # 1,C,H,W
            t = t / 255.0

            # 1) Upscale
            if abs(self.scale - 1.0) > 1e-3:
                t = F.interpolate(t, scale_factor=self.scale, mode="bicubic", align_corners=False)

            # 2) Blur
            blur = F.conv2d(t, self._kernel, padding=self._pad, groups=3)

            # 3) Unsharp
            sharp = t + self.amount * (t - blur)
            sharp = torch.clamp(sharp, 0.0, 1.0)

            # 4) Contrast around mean
            if abs(self.contrast - 1.0) > 1e-3:
                mean = sharp.mean(dim=(2, 3), keepdim=True)
                sharp = (sharp - mean) * self.contrast + mean
                sharp = torch.clamp(sharp, 0.0, 1.0)

            out = (sharp.squeeze(0).permute(1, 2, 0) * 255.0).to(torch.uint8)
            out = out.detach().cpu().numpy()
            return out
