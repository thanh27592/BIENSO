import re
import numpy as np
import cv2
from ultralytics import YOLO
import easyocr
import unicodedata
import torch
import os
import time
import logging
import threading
import inspect


LOG = logging.getLogger("anpr.alpr")

def pick_best_device():
    """
    Prefer best CUDA GPU (max VRAM). Fallback to CPU.
    Returns: ("cuda:N", N) or ("cpu", None)
    """
    if torch.cuda.is_available():
        best_i = 0
        best_mem = 0
        for i in range(torch.cuda.device_count()):
            try:
                mem = torch.cuda.get_device_properties(i).total_memory
            except Exception:
                mem = 0
            if mem > best_mem:
                best_mem = mem
                best_i = i
        return f"cuda:{best_i}", best_i
    return "cpu", None

# ===== Biển số VN bạn muốn =====
# 2 số - 1 chữ 1 số - 5 số  (vd 50A155555)
# 2 số - 2 chữ - 5 số       (vd 50AA55555)
# 2 số - 1 chữ - 5 số       (vd 50H55555)
VN_PATTERNS = [
    re.compile(r"^\d{2}[A-Z]\d\d{5}$"),     # 50A155555, 59V168055...
    re.compile(r"^\d{2}[A-Z]{2}\d{5}$"),    # 50AA58888
    re.compile(r"^\d{2}[A-Z]\d{5}$"),       # 60H00103, 50H77829
]

# alpr.py - THAY THẾ normalize_plate() hiện tại
def _strip_vn_diacritics(s: str) -> str:
    if not s:
        return ""
    # Đ/đ -> D
    s = s.replace("Đ", "D").replace("đ", "D")
    # tách dấu rồi bỏ dấu
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return s

def normalize_plate(text: str) -> str:
    """
    - Bỏ toàn bộ dấu tiếng Việt (sắc/huyền/hỏi/ngã/nặng)
    - Ô/Ơ/Ư -> O/O/U (do strip dấu)
    - Loại bỏ chữ W
    - Chỉ giữ A-Z0-9
    """
    t = _strip_vn_diacritics(str(text))
    t = t.upper()
    t = re.sub(r"[^A-Z0-9]", "", t)
    t = t.replace("W", "")  # loại W
    return t

def is_valid_vn_plate(t: str) -> bool:
    t = normalize_plate(t)
    return any(p.match(t) for p in VN_PATTERNS)

# ===== NEW: phân loại + validate xe máy / ô tô =====
# Template: N=digit, A=letter
MOTO_TEMPLATES = [
    ("MOTO_2A1N5N", "NNANNNNNN"),   # NN-XW NNNNN  => 2 số + 1 chữ + 1 số + 5 số (9)
    ("MOTO_2A1N4N", "NNANNNNN"),    # NN-XZ NNNN   => 2 số + 1 chữ + 1 số + 4 số (8)  (mới)
    ("MOTO_2A2A5N", "NNAANNNNN"),   # NN-XY NNNNN  => 2 số + 2 chữ + 5 số (9)
]

CAR_TEMPLATES = [
    ("CAR_2A4N", "NNANNNN"),        # NN-X NNNN    => 2 số + 1 chữ + 4 số (7)
    ("CAR_2A5N", "NNANNNNN"),       # NN-X NNNNN   => 2 số + 1 chữ + 5 số (8) (trùng pattern cũ nhưng giữ)
]

def _apply_ijl_fix_on_digit_positions(s: str, template: str) -> str:
    # chỉ đổi I/J/L -> 1 ở vị trí mà template yêu cầu là số (N)
    arr = list(s)
    for i, t in enumerate(template):
        if i >= len(arr):
            break
        if t == "N" and arr[i] in ("I", "J", "L"):
            arr[i] = "1"
    return "".join(arr)

def validate_and_canonicalize_plate(raw: str):
    """
    Trả về: (plate_canonical, kind) hoặc ("","")
    - plate_canonical: chuỗi đã normalize + fix I/J/L->1 theo vị trí số (cho xe máy)
    - kind: "MOTO" hoặc "CAR"
    """
    t = normalize_plate(raw)  # giữ toàn bộ filter cũ (bỏ dấu, chỉ A-Z0-9, remove W...)
    if not t:
        return "", ""

    # thử xe máy trước (vì format xe máy có thể dài và dễ nhầm)
    for _, tpl in MOTO_TEMPLATES:
        if len(t) != len(tpl):
            continue
        tt = _apply_ijl_fix_on_digit_positions(t, tpl)

        ok = True
        for i, ch in enumerate(tt):
            if tpl[i] == "N":
                if not ch.isdigit():
                    ok = False; break
            else:  # "A"
                if not ("A" <= ch <= "Z"):
                    ok = False; break

        if ok:
            # rule bạn yêu cầu: W/Z/Y không có I/J/L -> đã xử lý ở vị trí số.
            return tt, "MOTO"

    # ô tô
    for _, tpl in CAR_TEMPLATES:
        if len(t) != len(tpl):
            continue

        ok = True
        for i, ch in enumerate(t):
            if tpl[i] == "N":
                if not ch.isdigit():
                    ok = False; break
            else:
                if not ("A" <= ch <= "Z"):
                    ok = False; break

        if ok:
            return t, "CAR"

    # fallback: giữ nguyên validate cũ nếu bạn vẫn muốn hỗ trợ format cũ khác
    if is_valid_vn_plate(t):
        return t, "UNKNOWN"

    return "", ""

def _preprocess_for_ocr(crop_bgr: np.ndarray) -> np.ndarray:
    """Tăng khả năng OCR: upscale + sharpen + contrast"""
    if crop_bgr is None or crop_bgr.size == 0:
        return crop_bgr

    h, w = crop_bgr.shape[:2]

    # upscale nếu crop nhỏ
    # (biển số rõ thường cần chiều ngang crop >= ~220px trở lên)
    scale = 1
    if w < 220:
        scale = 3
    elif w < 320:
        scale = 2

    if scale > 1:
        crop_bgr = cv2.resize(crop_bgr, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    # tăng tương phản
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # sharpen nhẹ
    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    sharp = cv2.addWeighted(gray, 1.6, blur, -0.6, 0)

    # trả về RGB cho easyocr
    rgb = cv2.cvtColor(sharp, cv2.COLOR_GRAY2RGB)
    return rgb

def _preprocess_variants_for_ocr(crop_bgr: np.ndarray):
    """
    Trả về nhiều biến thể ảnh để OCR:
    - bản sharpen/CLAHE (cũ)
    - adaptive threshold (rất hợp biển VN 2 dòng)
    - otsu threshold
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return []

    # Variant 1: bản cũ (sharp gray -> RGB)
    v1 = _preprocess_for_ocr(crop_bgr)

    # Tạo gray chuẩn kích thước giống v1
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    scale = 1
    if w < 220:
        scale = 3
    elif w < 320:
        scale = 2
    if scale > 1:
        gray = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Variant 2: adaptive threshold
    th2 = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 10
    )
    # mở nét chữ
    th2 = cv2.morphologyEx(th2, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)

    # Variant 3: Otsu threshold
    _, th3 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    v2 = cv2.cvtColor(th2, cv2.COLOR_GRAY2RGB)
    v3 = cv2.cvtColor(th3, cv2.COLOR_GRAY2RGB)

    return [v1, v2, v3]


def _vn_score_candidate(txt: str, conf: float):
    """
    Chấm điểm candidate theo rule VN:
    - Ưu tiên validate_and_canonicalize_plate match MOTO/CAR
    - Ưu tiên có 2 số đầu
    - Ưu tiên độ dài hợp lý
    """
    from alpr import validate_and_canonicalize_plate  # tự gọi lại để tránh vòng import khi bạn refactor

    t = normalize_plate(txt)
    if not t or not re.match(r"^\d{2}", t):
        return -1e9, "", ""

    plate_can, kind = validate_and_canonicalize_plate(t)
    score = 0.0
    if plate_can:
        score += 100.0          # match template VN => điểm cực cao
        score += conf * 10.0
        score += len(plate_can) * 0.5
        return score, plate_can, kind

    # chưa match template, vẫn giữ để fallback (đừng finalize bằng nó ở camera_worker)
    score += 10.0
    score += conf * 10.0
    score += len(t) * 0.2
    return score, t, ""

def _cluster_lines(ocr_items):
    """
    ocr_items: list of (bbox, text, conf)
    Group theo dòng dựa trên y-center.
    """
    if not ocr_items:
        return []

    items = []
    for bbox, txt, conf in ocr_items:
        # bbox = [[x,y], [x,y], [x,y], [x,y]]
        ys = [p[1] for p in bbox]
        xs = [p[0] for p in bbox]
        y_center = sum(ys) / 4.0
        x_center = sum(xs) / 4.0
        h = (max(ys) - min(ys)) + 1e-6
        items.append((y_center, x_center, h, txt, conf))

    items.sort(key=lambda z: (z[0], z[1]))

    # ngưỡng gom dòng: dựa vào chiều cao chữ
    lines = []
    cur = []
    cur_y = None
    cur_h = None

    for y, x, h, txt, conf in items:
        if cur_y is None:
            cur = [(x, txt, conf)]
            cur_y = y
            cur_h = h
            continue

        # nếu lệch y ít => cùng dòng
        if abs(y - cur_y) <= 0.6 * max(cur_h, h):
            cur.append((x, txt, conf))
            # update trung bình
            cur_y = (cur_y + y) / 2.0
            cur_h = (cur_h + h) / 2.0
        else:
            # kết thúc dòng cũ
            cur.sort(key=lambda t: t[0])
            lines.append(cur)
            # mở dòng mới
            cur = [(x, txt, conf)]
            cur_y = y
            cur_h = h

    if cur:
        cur.sort(key=lambda t: t[0])
        lines.append(cur)

    return lines

class ALPR:
    def __init__(self, model_path: str, conf_thres: float = 0.35):
        self.model = YOLO(model_path)
        self.conf_thres = conf_thres

        self.use_cuda = bool(torch.cuda.is_available())
        if self.use_cuda and torch.cuda.device_count() > 0:
            # chọn GPU có VRAM lớn nhất
            best = max(
                range(torch.cuda.device_count()),
                key=lambda i: torch.cuda.get_device_properties(i).total_memory
            )
            torch.cuda.set_device(best)
            self.device = best
        else:
            self.device = "cpu"
            self.use_cuda = False

        # easyocr dùng cuda nếu có
        self.reader = easyocr.Reader(["en"], gpu=self.use_cuda)

        # serialize inference calls across threads (YOLO + easyocr) to avoid GPU/thread issues
        self._infer_lock = threading.Lock()

        # easyocr readtext kwargs compatibility (different easyocr versions)
        try:
            self._rt_params = set(inspect.signature(self.reader.readtext).parameters.keys())
        except Exception:
            self._rt_params = set()
        self._ocr_allowlist = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self._last_ocr_err_ts = 0.0

    def _easyocr_readtext(self, img, paragraph: bool = False):
        """
        Wrapper for easyocr.Reader.readtext with:
          - thread lock (GPU safety)
          - allowlist for VN plates
          - compatibility with older easyocr versions (missing kwargs)
          - throttled exception logging (avoid spam)
        Returns: list of OCR items (detail=1) or [] on failure.
        """
        kwargs = {}
        if "paragraph" in getattr(self, "_rt_params", set()):
            kwargs["paragraph"] = bool(paragraph)
        if "allowlist" in getattr(self, "_rt_params", set()):
            kwargs["allowlist"] = getattr(self, "_ocr_allowlist", "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")

        try:
            with self._infer_lock:
                return self.reader.readtext(img, detail=1, **kwargs)
        except TypeError as e:
            # likely unsupported kwargs in older easyocr
            now = time.time()
            if now - float(getattr(self, "_last_ocr_err_ts", 0.0)) > 2.0:
                LOG.warning(f"[easyocr] readtext TypeError -> retry without kwargs. err={e}")
                self._last_ocr_err_ts = now
            try:
                with self._infer_lock:
                    return self.reader.readtext(img, detail=1)
            except Exception:
                now = time.time()
                if now - float(getattr(self, "_last_ocr_err_ts", 0.0)) > 2.0:
                    LOG.exception("[easyocr] readtext failed (no kwargs)")
                    self._last_ocr_err_ts = now
                return []
        except Exception:
            now = time.time()
            if now - float(getattr(self, "_last_ocr_err_ts", 0.0)) > 2.0:
                LOG.exception("[easyocr] readtext failed")
                self._last_ocr_err_ts = now
            return []

    def _extract_candidates_from_ocr(self, res):
        """Convert easyocr results to list[(text, conf)]. Handles 2-line VN plates."""
        candidates: list[tuple[str, float]] = []
        if not res:
            return candidates

        # some setups / versions may return list[str]
        if isinstance(res, (list, tuple)) and res and isinstance(res[0], str):
            for s in res:
                if isinstance(s, str) and s.strip():
                    candidates.append((s.strip(), 0.0))
            return candidates

        items = []
        for item in res:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            bbox = item[0] if len(item) >= 1 else None
            txt = str(item[1] or "").strip() if len(item) >= 2 else ""
            conf = 0.0
            if len(item) >= 3:
                try:
                    conf = float(item[2])
                except Exception:
                    conf = 0.0
            if txt:
                items.append((bbox, txt, conf))

        if not items:
            return candidates

        # group by lines (2-line plates)
        lines = []
        try:
            lines = _cluster_lines([(b, t, c) for (b, t, c) in items if b is not None])
        except Exception:
            lines = []

        if lines:
            line_texts: list[str] = []
            line_confs: list[float] = []
            for line in lines:
                t = "".join(seg[1] for seg in line)
                if line:
                    c = float(sum(seg[2] for seg in line) / len(line))
                else:
                    c = 0.0
                if t:
                    line_texts.append(t)
                    line_confs.append(c)

            for t, c in zip(line_texts, line_confs):
                candidates.append((t, float(c)))

            if len(line_texts) >= 2:
                candidates.append((line_texts[0] + line_texts[1], float(min(line_confs[0], line_confs[1]))))
            if len(line_texts) > 2:
                candidates.append(("".join(line_texts), float(min(line_confs))))
            return candidates

        # fallback: sort by x-center and join
        try:
            tmp = []
            for bbox, txt, conf in items:
                if bbox is None:
                    continue
                xs = [p[0] for p in bbox] if isinstance(bbox, (list, tuple)) else []
                x_center = float(sum(xs) / 4.0) if xs else 0.0
                tmp.append((x_center, txt, conf))
            tmp.sort(key=lambda x: x[0])
            joined = "".join(x[1] for x in tmp)
            conf_best = float(max((x[2] for x in tmp), default=0.0))
            if joined:
                candidates.append((joined, conf_best))
        except Exception:
            pass

        return candidates

    def detect_plate_boxes(self, frame_bgr: np.ndarray, roi=None):
        """
        roi: (x1,y1,x2,y2) nếu muốn detect trong vùng chuyển động.
        Trả về list: [((x1,y1,x2,y2), conf), ...] theo toạ độ frame gốc.
        """
        if roi is not None:
            rx1, ry1, rx2, ry2 = roi
            crop = frame_bgr[ry1:ry2, rx1:rx2]
            if crop.size == 0:
                return []
            with self._infer_lock:
                results = self.model.predict(crop, conf=self.conf_thres, verbose=False, device=self.device)
        else:
            with self._infer_lock:
                results = self.model.predict(frame_bgr, conf=self.conf_thres, verbose=False, device=self.device)

        if not results:
            return []
        r = results[0]
        if r.boxes is None:
            return []

        out = []
        for b in r.boxes:
            xyxy = b.xyxy[0].cpu().numpy().tolist()
            conf = float(b.conf[0].cpu().numpy())

            x1, y1, x2, y2 = map(int, xyxy)
            if roi is not None:
                # cộng bù về toạ độ gốc
                x1 += rx1; x2 += rx1
                y1 += ry1; y2 += ry1

            out.append(((x1, y1, x2, y2), conf))
        return out
    
    def _expand_bbox(self, x1, y1, x2, y2, w, h, mx_ratio=0.10, my_ratio=0.18):
        bw = max(1, (x2 - x1))
        bh = max(1, (y2 - y1))

        mx = int(mx_ratio * bw)
        my = int(my_ratio * bh)

        # nới TOP mạnh hơn để bắt đủ dòng trên của biển 2 dòng
        top = int(my * 1.8)
        bot = int(my * 0.8)

        x1 = max(0, x1 - mx)
        x2 = min(w, x2 + mx)
        y1 = max(0, y1 - top)
        y2 = min(h, y2 + bot)
        return x1, y1, x2, y2
    
    def detect_multi(self, frame_bgr: np.ndarray, max_plates: int = 2, roi=None):
        """
        Chỉ detect bbox + crop, KHÔNG OCR.
        Trả về list: [{"det_conf","bbox","crop"}...]

        FIX (multi-object):
        - Không chọn theo area đơn thuần (dễ bị "2 box cùng 1 biển" ăn hết slot, mất biển khác)
        - Áp dụng NMS (loại bỏ box trùng) rồi chọn theo confidence (ưu tiên đúng đối tượng)
        """
        h, w = frame_bgr.shape[:2]
        boxes = self.detect_plate_boxes(frame_bgr, roi=roi)
        if not boxes:
            return []

        def _iou(bb1, bb2):
            x1, y1, x2, y2 = bb1
            X1, Y1, X2, Y2 = bb2
            ix1 = max(x1, X1)
            iy1 = max(y1, Y1)
            ix2 = min(x2, X2)
            iy2 = min(y2, Y2)
            iw = max(0, ix2 - ix1)
            ih = max(0, iy2 - iy1)
            inter = iw * ih
            if inter <= 0:
                return 0.0
            a = max(0, x2 - x1) * max(0, y2 - y1)
            b = max(0, X2 - X1) * max(0, Y2 - Y1)
            den = a + b - inter
            return float(inter / den) if den > 0 else 0.0

        # NMS threshold: giảm trùng box (có thể chỉnh qua env)
        try:
            nms_iou = float(os.getenv("ANPR_DET_NMS_IOU", "0.50"))
        except Exception:
            nms_iou = 0.50


        # Extra filters to reduce huge false-positive boxes (tunable via env)
        # - If your camera is very close-up and plates take a big portion of the frame,
        #   you can relax these ratios.
        try:
            det_min_area = float(os.getenv("ANPR_DET_MIN_AREA", "700"))
        except Exception:
            det_min_area = 700.0
        try:
            det_max_w_ratio = float(os.getenv("ANPR_DET_MAX_W_RATIO", "0.65"))
        except Exception:
            det_max_w_ratio = 0.65
        try:
            det_max_h_ratio = float(os.getenv("ANPR_DET_MAX_H_RATIO", "0.55"))
        except Exception:
            det_max_h_ratio = 0.55
        try:
            det_ar_min = float(os.getenv("ANPR_DET_AR_MIN", "0.75"))
        except Exception:
            det_ar_min = 0.75
        try:
            det_ar_max = float(os.getenv("ANPR_DET_AR_MAX", "6.80"))
        except Exception:
            det_ar_max = 6.80

        cand = []
        for (x1, y1, x2, y2), conf in boxes:
            bw = max(0, x2 - x1)
            bh = max(0, y2 - y1)
            area = bw * bh

            # 1) tiny boxes are usually noise
            if area < det_min_area:
                continue
            if bw <= 1 or bh <= 1:
                continue

            # 2) aspect ratio filter: VN plates are usually wide, but moto 2-line can be closer to square
            ar = float(bw) / float(bh)
            if ar < det_ar_min or ar > det_ar_max:
                continue

            # 3) reject unrealistically huge boxes (common false-positive)
            if bw > int(det_max_w_ratio * w) or bh > int(det_max_h_ratio * h):
                continue
            # bỏ vùng OSD góc dưới phải (giữ logic cũ)
            if x1 > int(0.72 * w) and y1 > int(0.72 * h):
                continue
            cand.append(((int(x1), int(y1), int(x2), int(y2)), float(conf), float(area)))

        if not cand:
            return []

        # sort theo conf trước, area sau
        cand.sort(key=lambda t: (t[1], t[2]), reverse=True)

        kept = []
        for bb, conf, area in cand:
            dup = False
            for kbb, kconf, karea in kept:
                if _iou(bb, kbb) >= nms_iou:
                    dup = True
                    break
            if dup:
                continue
            kept.append((bb, conf, area))
            if len(kept) >= int(max_plates):
                break

        out = []
        for (x1, y1, x2, y2), det_conf, _ in kept:
            x1, y1, x2, y2 = self._expand_bbox(x1, y1, x2, y2, w, h)
            crop = frame_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            out.append({"det_conf": float(det_conf), "bbox": (x1, y1, x2, y2), "crop": crop})

        return out

    def detect_and_read_multi(self, frame_bgr: np.ndarray, max_plates: int = 2, roi=None):
        """
        Trả về list kết quả (tối đa max_plates):
        [
          {"plate","det_conf","ocr_conf","bbox","crop"}
        ]
        """
        h, w = frame_bgr.shape[:2]
        boxes = self.detect_plate_boxes(frame_bgr, roi=roi)
        if not boxes:
            return []

        # lọc + sắp theo area desc
        filtered = []
        for (x1, y1, x2, y2), conf in boxes:
            bw = max(0, x2 - x1)
            bh = max(0, y2 - y1)
            area = bw * bh
            if area < 900:
                continue

            # bỏ vùng OSD góc dưới phải (giữ logic cũ của bạn)
            if x1 > int(0.72 * w) and y1 > int(0.72 * h):
                continue

            filtered.append((area, (x1, y1, x2, y2), float(conf)))

        if not filtered:
            return []

        filtered.sort(key=lambda z: z[0], reverse=True)
        filtered = filtered[:max_plates]

        out = []
        for _, (x1, y1, x2, y2), det_conf in filtered:
            x1, y1, x2, y2 = self._expand_bbox(x1, y1, x2, y2, w, h)
            crop = frame_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            plate_text, ocr_conf = self.read_plate_text(crop)
            if not plate_text:
                continue

            out.append({
                "plate": plate_text,
                "det_conf": det_conf,
                "ocr_conf": float(ocr_conf),
                "bbox": (x1, y1, x2, y2),
                "crop": crop
            })
        return out

    def read_plate_text(self, crop_bgr: np.ndarray, fast: bool = True) -> tuple[str, float]:
        """
        OCR robust cho biển VN.
        - Hỗ trợ biển 2 dòng (xe máy) bằng cách gom OCR theo dòng rồi nối lại.
        - Dùng allowlist để giảm nhiễu.
        - Tự tương thích easyocr cũ (không hỗ trợ một số kwargs như paragraph/allowlist).

        Return: (best_text, best_conf)
          - Nếu tìm được biển số canonical (đúng format VN) -> trả về canonical.
          - Nếu không -> trả về text normalize tốt nhất để bạn nhìn log xem OCR đang đọc ra gì.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return "", 0.0

        # upscale để OCR dễ đọc
        h, w = crop_bgr.shape[:2]
        if max(h, w) < 130:
            scale = 3.0
        elif max(h, w) < 220:
            scale = 2.0
        else:
            scale = 1.5

        crop_up = cv2.resize(
            crop_bgr,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_CUBIC,
        )

        variants: list[tuple[str, np.ndarray]] = [("up", crop_up)]

        if not fast:
            # preprocess: gray + bilateral + adaptive threshold (2 biến thể: trắng nền / đen nền)
            try:
                gray = cv2.cvtColor(crop_up, cv2.COLOR_BGR2GRAY)
                gray = cv2.bilateralFilter(gray, 9, 75, 75)
                thr = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, 31, 7
                )
                variants.append(("thr", thr))
                variants.append(("thr_inv", cv2.bitwise_not(thr)))
            except Exception:
                pass

            # sharpen nhẹ
            try:
                blur = cv2.GaussianBlur(crop_up, (0, 0), 1.0)
                sharp = cv2.addWeighted(crop_up, 1.5, blur, -0.5, 0)
                variants.append(("sharp", sharp))
            except Exception:
                pass

        best_plate = ""
        best_conf = 0.0

        # debug visibility: nếu không ra canonical, vẫn trả về "best_raw" để log biết OCR thấy gì
        best_raw = ""
        best_raw_conf = 0.0

        for tag, img in variants:
            img_in = img
            # easyocr thường ổn với RGB; convert BGR->RGB để tránh khác phiên bản
            try:
                if isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[2] == 3:
                    img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except Exception:
                img_in = img

            res = self._easyocr_readtext(img_in, paragraph=False)
            cands = self._extract_candidates_from_ocr(res)

            for txt, conf in cands:
                txt_norm = normalize_plate(txt)
                if not txt_norm:
                    continue

                # keep raw best for debug (even if not canonical)
                if (conf > best_raw_conf) or (conf == best_raw_conf and len(txt_norm) > len(best_raw)):
                    best_raw = txt_norm
                    best_raw_conf = float(conf)

                plate_can, _kind = validate_and_canonicalize_plate(txt_norm)
                if not plate_can:
                    continue

                # scoring: ưu tiên conf + độ dài
                score = float(conf) * 10.0 + float(len(plate_can))
                cur_score = float(best_conf) * 10.0 + float(len(best_plate)) if best_plate else -1e9
                if (not best_plate) or (score > cur_score):
                    best_plate = plate_can
                    best_conf = float(conf)

            # fast mode: nếu đã ra canonical khá chắc -> stop early để nhanh
            if fast and best_plate and best_conf >= 0.55:
                break

        if best_plate:
            return best_plate, float(best_conf)

        return best_raw, float(best_raw_conf)

    def detect_and_read(self, frame_bgr: np.ndarray):
        boxes = self.detect_plate_boxes(frame_bgr)
        if not boxes:
            return None

        h, w = frame_bgr.shape[:2]

        # chọn bbox lớn nhất nhưng cũng phải hợp lý (lọc vùng OSD "Camera 01" ở góc dưới phải)
        best = None
        best_area = 0

        for (xyxy, conf) in boxes:
            x1, y1, x2, y2 = map(int, xyxy)
            bw = max(0, x2 - x1)
            bh = max(0, y2 - y1)
            area = bw * bh
            if area <= 0:
                continue

            # bỏ vùng OSD hay nằm góc dưới phải (tùy camera, bạn có thể chỉnh)
            if x1 > int(0.72 * w) and y1 > int(0.72 * h):
                continue

            # bỏ bbox quá nhỏ
            if area < 900:
                continue

            if area > best_area:
                best_area = area
                best = (x1, y1, x2, y2, float(conf))

        if best is None:
            return None

        x1, y1, x2, y2, det_conf = best

        # nới margin
        mx = int(0.10 * (x2 - x1))
        my = int(0.22 * (y2 - y1))
        x1 = max(0, x1 - mx); y1 = max(0, y1 - my)
        x2 = min(w, x2 + mx); y2 = min(h, y2 + my)

        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        plate_text, ocr_conf = self.read_plate_text(crop)
        if not plate_text:
            return None

        return {
            "plate": plate_text,
            "det_conf": det_conf,
            "ocr_conf": float(ocr_conf),
            "bbox": (x1, y1, x2, y2),
            "crop": crop
        }
