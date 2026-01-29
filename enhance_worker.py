# enhance_worker.py
# CPU enhancement worker for license plate crops.
# Runs in a separate thread, does heavier preprocessing + OCR, and returns best candidate
# so the realtime loop stays responsive.

from __future__ import annotations

import os
import cv2
import numpy as np
import time
import queue
import threading
import logging

LOG = logging.getLogger("anpr.enhance")

# Optional GPU enhancement (torch-based). If not available, we fall back to CPU pipeline.
try:
    from gpu_enhance import TorchPlateEnhancer
except Exception:
    TorchPlateEnhancer = None  # type: ignore


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "y", "on")


def _env_float(name: str, default: str) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return float(default)


def _env_str(name: str, default: str) -> str:
    try:
        return str(os.environ.get(name, default))
    except Exception:
        return str(default)


def _laplacian_var(gray: np.ndarray) -> float:
    try:
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception:
        return 0.0


def enhance_plate_bgr(crop_bgr: np.ndarray) -> np.ndarray:
    """
    Enhance crop biển số (CPU).
    Pipeline:
      - upscale 2-3x (INTER_CUBIC)
      - CLAHE tăng tương phản (kênh L)
      - unsharp mask làm nét
      - denoise nhẹ
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return crop_bgr

    h, w = crop_bgr.shape[:2]
    scale = 3.0 if max(h, w) < 140 else 2.0
    up = cv2.resize(crop_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    # CLAHE (LAB)
    lab = cv2.cvtColor(up, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    up2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    # Unsharp mask (tăng nét)
    blur = cv2.GaussianBlur(up2, (0, 0), 1.1)
    sharp = cv2.addWeighted(up2, 1.6, blur, -0.6, 0)

    # Denoise nhẹ
    out = cv2.fastNlMeansDenoisingColored(sharp, None, 3, 3, 7, 21)
    return out


class EnhanceWorker:
    """
    Thread xử lý enhance + OCR lại (slow path).
    - submit(...) non-blocking; nếu queue đầy thì bỏ qua để không ảnh hưởng realtime.
    - poll() lấy kết quả (0..n item).
    """

    def __init__(self, alpr, max_queue: int = 12):
        self.alpr = alpr
        self.in_q: "queue.Queue[tuple[float, str, str, int, np.ndarray]]" = queue.Queue(maxsize=max_queue)
        self.out_q: "queue.Queue[tuple[str, str, int, str, float, np.ndarray, float]]" = queue.Queue(maxsize=max_queue)
        self._run = True
        self._t = threading.Thread(target=self._loop, daemon=True, name="EnhanceWorker")

        # Optional GPU enhancement parameters
        self._gpu_enable = _env_flag("ANPR_GPU_ENHANCE", "0")
        self._gpu_mode = _env_str("ANPR_GPU_ENHANCE_MODE", "torch").strip().lower()  # torch | cpu+torch
        self._torch_enhancer = None

        if self._gpu_enable and TorchPlateEnhancer is not None:
            try:
                scale = _env_float("ANPR_GPU_ENHANCE_SCALE", "2.0")
                amount = _env_float("ANPR_GPU_ENHANCE_AMOUNT", "1.0")
                contrast = _env_float("ANPR_GPU_ENHANCE_CONTRAST", "1.15")
                self._torch_enhancer = TorchPlateEnhancer(scale=scale, amount=amount, contrast=contrast)
                LOG.info(f"[EnhanceWorker] GPU enhancer enabled (mode={self._gpu_mode} scale={scale} amount={amount} contrast={contrast})")
            except Exception:
                LOG.exception("[EnhanceWorker] failed to init TorchPlateEnhancer -> fallback CPU")
                self._torch_enhancer = None
        elif self._gpu_enable and TorchPlateEnhancer is None:
            LOG.warning("[EnhanceWorker] ANPR_GPU_ENHANCE=1 but gpu_enhance.TorchPlateEnhancer not available -> CPU")

    def start(self):
        LOG.info("[EnhanceWorker] started")
        self._t.start()

    def stop(self):
        self._run = False

    def submit(self, focus_id: str, cam_name: str, zone_idx: int, crop_bgr: np.ndarray):
        item = (time.time(), focus_id, cam_name, int(zone_idx), crop_bgr)
        try:
            self.in_q.put_nowait(item)
            return
        except queue.Full:
            # Drop oldest to keep latest frames (commercial CCTV style)
            try:
                _ = self.in_q.get_nowait()
            except queue.Empty:
                pass
            try:
                self.in_q.put_nowait(item)
            except queue.Full:
                LOG.debug(f"[EnhanceWorker] in_q full -> drop newest focus_id={focus_id} cam={cam_name} zone={zone_idx}")

    def _enhance(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Enhance crop. Optionally uses torch GPU. Always returns BGR image."""
        if crop_bgr is None or crop_bgr.size == 0:
            return crop_bgr

        if self._torch_enhancer is not None:
            try:
                if self._gpu_mode == "cpu+torch":
                    return self._torch_enhancer.enhance(enhance_plate_bgr(crop_bgr))
                return self._torch_enhancer.enhance(crop_bgr)
            except Exception:
                LOG.exception("[EnhanceWorker] torch enhance failed -> fallback CPU")

        return enhance_plate_bgr(crop_bgr)

    def poll(self):
        items = []
        while True:
            try:
                items.append(self.out_q.get_nowait())
            except queue.Empty:
                break
        return items

    def _loop(self):
        while self._run:
            try:
                _, focus_id, cam, zone, crop_bgr = self.in_q.get(timeout=0.2)
            except queue.Empty:
                continue

            t0 = time.time()
            try:
                enh = self._enhance(crop_bgr)

                g = cv2.cvtColor(enh, cv2.COLOR_BGR2GRAY)
                sharp = _laplacian_var(g)

                # OCR slow path
                plate_raw, conf = self.alpr.read_plate_text(enh, fast=False)

                try:
                    self.out_q.put_nowait((focus_id, cam, zone, plate_raw, float(conf), enh, float(sharp)))
                except queue.Full:
                    LOG.debug(f"[EnhanceWorker] out_q full -> drop result focus_id={focus_id} cam={cam} zone={zone}")

                dt_ms = (time.time() - t0) * 1000.0
                LOG.debug(f"[EnhanceWorker] done focus_id={focus_id} cam={cam} zone={zone} plate='{plate_raw}' conf={float(conf):.2f} sharp={float(sharp):.1f} ({dt_ms:.1f}ms)")
            except Exception as e:
                LOG.exception(f"[EnhanceWorker] error: {type(e).__name__}: {e}")
                continue
