# camera_worker.py
from __future__ import annotations

import os
import time
import math
import threading
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from ui import bgr_to_qimage
from enhance_worker import EnhanceWorker


def _laplacian_var(bgr: np.ndarray) -> float:
    """Simple sharpness metric (higher = sharper)."""
    try:
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(g, cv2.CV_64F).var())
    except Exception:
        return 0.0


def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    b_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = a_area + b_area - inter
    return float(inter / denom) if denom > 0 else 0.0


class CameraWorker(QThread):
    """
    Realtime camera worker:
    - Thread 1 (QThread.run): đọc RTSP + emit frame ra UI (có overlay bbox track)
    - Thread 2 (_process_loop): detect plate -> track theo IOU (khái niệm "1 đối tượng")
                               -> chụp nhiều crop -> chọn best-shot -> OCR vote
                               -> emit UPSERT event (insert lần đầu, update nếu ảnh sau tốt hơn)
    """
    frame_ready = Signal(str, object)              # (camera_name, QImage)
    focus_ready = Signal(str, int, object)         # (camera_name, slot_idx, QImage)
    plate_event = Signal(dict)                     # payload dict (UPSERT / FINAL)
    status = Signal(str, str)                      # (camera_name, status)

    def __init__(
        self,
        camera_name: str,
        role: str,
        rtsp: str,
        alpr,
        snapshot_dir: str,
        debounce_sec: int = 12,
        detect_every_n: int = 2,
    ):
        super().__init__()
        self.camera_name = camera_name
        self.role = role
        self.rtsp = rtsp
        self.alpr = alpr
        self.snapshot_dir = snapshot_dir
        self.debounce_sec = int(debounce_sec)
        self.detect_every_n = int(detect_every_n)

        self._running = True
        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_epoch = 0.0

        # overlay boxes (process thread -> run thread)
        self._overlay_lock = threading.Lock()
        self._overlay_boxes: List[dict] = []

        self._proc_thread: Optional[threading.Thread] = None
        self._enhancer: Optional[EnhanceWorker] = None

    def stop(self):
        self._running = False
        try:
            if self._enhancer:
                self._enhancer.stop()
        except Exception:
            pass

    def run(self):
        cap = cv2.VideoCapture(self.rtsp, cv2.CAP_FFMPEG)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        self.status.emit(self.camera_name, "RUNNING")

        # slow-path enhancement + OCR worker
        self._enhancer = EnhanceWorker(self.alpr, max_queue=24)
        self._enhancer.start()

        # processing loop (detect + track)
        self._proc_thread = threading.Thread(target=self._process_loop, name=f"proc-{self.camera_name}", daemon=True)
        self._proc_thread.start()

        while self._running:
            ok, frame = cap.read()
            if not ok or frame is None:
                self.status.emit(self.camera_name, "READ FAILED - reconnecting...")
                cap.release()
                time.sleep(1.5)
                cap = cv2.VideoCapture(self.rtsp, cv2.CAP_FFMPEG)
                continue

            # downscale for speed
            h, w = frame.shape[:2]
            target_w = 1920
            if w > target_w:
                scale = target_w / w
                frame = cv2.resize(frame, (target_w, int(h * scale)), interpolation=cv2.INTER_AREA)

            # store raw frame for process loop
            with self._frame_lock:
                self._latest_frame = frame
                self._latest_epoch = time.time()

            # draw overlay for UI (do NOT write into stored frame)
            disp = frame.copy()
            with self._overlay_lock:
                boxes = list(self._overlay_boxes)

            for b in boxes:
                try:
                    x1, y1, x2, y2 = b.get("bbox", (0, 0, 0, 0))
                    color = b.get("color", (255, 0, 0))  # blue in BGR
                    label = b.get("label", "")
                    cv2.rectangle(disp, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    if label:
                        y_text = max(15, int(y1) - 6)
                        cv2.putText(
                            disp,
                            str(label),
                            (int(x1), y_text),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            2,
                            cv2.LINE_AA,
                        )
                except Exception:
                    pass

            self.frame_ready.emit(self.camera_name, bgr_to_qimage(disp))

        cap.release()
        self.status.emit(self.camera_name, "STOPPED")

    def _process_loop(self):
        from alpr import validate_and_canonicalize_plate

        # ===== tuning =====
        # ===== tuning =====
        ui_slots = int(os.getenv("ANPR_UI_SLOTS", "2"))  # UI focus slots (default 2)
        max_tracks = int(os.getenv("ANPR_MAX_TRACKS", "6"))   # tổng số track được theo dõi đồng thời
        # số plate detections lấy mỗi vòng (>= max_tracks để tránh '2 box cùng 1 biển' ăn hết slot)
        max_plate_dets = int(os.getenv("ANPR_MAX_PLATE_DETS", str(max(8, max_tracks * 2))))

        iou_match_thres = float(os.getenv("ANPR_TRACK_IOU", "0.22"))
        inside_match_score = float(os.getenv("ANPR_INSIDE_MATCH_SCORE", "0.12"))
        duplicate_det_iou = float(os.getenv("ANPR_DUPLICATE_DET_IOU", "0.65"))
        track_lost_sec = float(os.getenv("ANPR_TRACK_LOST_SEC", "2.5"))
        min_track_age_to_commit = float(os.getenv("ANPR_MIN_TRACK_AGE", "0.35"))
        vote_need = int(os.getenv("ANPR_VOTE_NEED", "2"))
        min_final_conf = float(os.getenv("ANPR_MIN_FINAL_CONF", "0.62"))
        high_conf_one_shot = float(os.getenv("ANPR_HIGH_CONF_ONE_SHOT", "0.78"))
        update_score_delta = float(os.getenv("ANPR_UPDATE_SCORE_DELTA", "0.07"))
        update_min_interval = float(os.getenv("ANPR_UPDATE_MIN_INTERVAL", "0.45"))
        enhance_trigger_conf = float(os.getenv("ANPR_ENHANCE_TRIGGER_CONF", "0.55"))
        sharp_trigger = float(os.getenv("ANPR_SHARP_TRIGGER", "85"))
        enhance_min_interval = float(os.getenv("ANPR_ENHANCE_MIN_INTERVAL", "0.25"))
        max_pend = int(os.getenv("ANPR_MAX_PEND", "8"))

        # Force a full-frame plate detection every N seconds (even if motion_roi is active)
        # This helps in scenes with 2+ objects where motion_roi may only cover one area.
        fullframe_sec = float(os.getenv("ANPR_FULLFRAME_SEC", "1.0"))

        # Association bbox: used ONLY for matching when IOU is low (plate bbox jitter).
        # Keep it small to avoid merging 2 nearby objects into 1 track.
        assoc_expand_x = float(os.getenv("ANPR_ASSOC_EXPAND_X", "0.9"))
        assoc_expand_up = float(os.getenv("ANPR_ASSOC_EXPAND_UP", "2.2"))
        assoc_expand_down = float(os.getenv("ANPR_ASSOC_EXPAND_DOWN", "1.4"))
        assoc_max_dist_ratio = float(os.getenv("ANPR_ASSOC_MAX_DIST_RATIO", "3.0"))
        assoc_max_dist_cap = float(os.getenv("ANPR_ASSOC_MAX_DIST_CAP", "260"))

        draw_motion_roi = bool(int(os.getenv("ANPR_DRAW_MOTION_ROI", "1")))

        # object-box expansion (to look more like commercial CCTV tracking)
        obj_expand_x = float(os.getenv("ANPR_OBJ_EXPAND_X", "2.2"))
        obj_expand_up = float(os.getenv("ANPR_OBJ_EXPAND_UP", "6.5"))
        obj_expand_down = float(os.getenv("ANPR_OBJ_EXPAND_DOWN", "2.0"))

        # clamp object-box size to avoid 2 objects being merged into 1 huge box
        obj_max_w_ratio = float(os.getenv("ANPR_OBJ_MAX_W_RATIO", "0.85"))
        obj_max_h_ratio = float(os.getenv("ANPR_OBJ_MAX_H_RATIO", "0.90"))

        def _bbox_center(bb):
            x1, y1, x2, y2 = bb
            return int((x1 + x2) * 0.5), int((y1 + y2) * 0.5)

        def _point_in_bb(pt, bb):
            cx, cy = pt
            x1, y1, x2, y2 = bb
            return (x1 <= cx <= x2) and (y1 <= cy <= y2)

        def _expand_obj_bbox(bb_plate, frame_shape, kind_hint=""):
            h0, w0 = frame_shape[:2]
            x1, y1, x2, y2 = map(int, bb_plate)
            pw = max(1, x2 - x1)
            ph = max(1, y2 - y1)
            kh = (kind_hint or "").upper()

            if kh == "MOTO":
                ex = int(pw * max(obj_expand_x, 2.4))
                up = int(ph * max(obj_expand_up, 7.5))
                down = int(ph * max(obj_expand_down, 2.0))
            elif kh == "CAR":
                ex = int(pw * max(obj_expand_x, 2.0))
                up = int(ph * max(obj_expand_up, 6.0))
                down = int(ph * max(obj_expand_down, 2.0))
            else:
                ex = int(pw * obj_expand_x)
                up = int(ph * obj_expand_up)
                down = int(ph * obj_expand_down)

            ox1 = max(0, x1 - ex)
            ox2 = min(w0 - 1, x2 + ex)
            oy1 = max(0, y1 - up)
            oy2 = min(h0 - 1, y2 + down)

            if ox2 <= ox1:
                ox2 = min(w0 - 1, ox1 + 1)
            if oy2 <= oy1:
                oy2 = min(h0 - 1, oy1 + 1)
            

            # clamp overly huge boxes (visual + matching stability)
            try:
                max_w = int(w0 * float(obj_max_w_ratio))
                max_h = int(h0 * float(obj_max_h_ratio))
                cur_w = int(ox2 - ox1)
                cur_h = int(oy2 - oy1)
                if max_w > 10 and cur_w > max_w:
                    cx = int((x1 + x2) * 0.5)
                    ox1 = max(0, cx - max_w // 2)
                    ox2 = min(w0 - 1, ox1 + max_w)
                if max_h > 10 and cur_h > max_h:
                    cy = int((y1 + y2) * 0.5)
                    oy1 = max(0, cy - max_h // 2)
                    oy2 = min(h0 - 1, oy1 + max_h)
            except Exception:
                pass
            return (int(ox1), int(oy1), int(ox2), int(oy2))

        def _expand_assoc_bbox(bb_plate, frame_shape):
            """A small association bbox around plate bbox (for jitter-tolerant matching).

            IMPORTANT: This is NOT the big CCTV-like object bbox. It is intentionally small
            to prevent 2 objects being merged when they are close together.
            """
            h0, w0 = frame_shape[:2]
            x1, y1, x2, y2 = map(int, bb_plate)
            pw = max(1, x2 - x1)
            ph = max(1, y2 - y1)

            ex = int(pw * assoc_expand_x)
            up = int(ph * assoc_expand_up)
            down = int(ph * assoc_expand_down)

            ax1 = max(0, x1 - ex)
            ax2 = min(w0 - 1, x2 + ex)
            ay1 = max(0, y1 - up)
            ay2 = min(h0 - 1, y2 + down)

            if ax2 <= ax1:
                ax2 = min(w0 - 1, ax1 + 1)
            if ay2 <= ay1:
                ay2 = min(h0 - 1, ay1 + 1)
            return (int(ax1), int(ay1), int(ax2), int(ay2))

        def _assoc_max_dist(bb_plate):
            x1, y1, x2, y2 = map(int, bb_plate)
            pw = max(1, x2 - x1)
            ph = max(1, y2 - y1)
            # dynamic distance gate: proportional to plate bbox size
            return float(min(assoc_max_dist_cap, max(80.0, assoc_max_dist_ratio * max(pw, ph))))


        # motion ROI
        prev_gray = None
        motion_roi = None
        motion_hold_until = 0.0
        frame_idx = 0
        last_fullframe_detect_ts = 0.0

        # track states
        tracks: Dict[int, dict] = {}
        focus_to_tid: Dict[str, int] = {}
        slot_to_tid: Dict[int, int] = {}   # ui slot -> track id
        next_tid = 1

        def _alloc_slot() -> Optional[int]:
            for s in range(ui_slots):
                if s not in slot_to_tid:
                    return s
            return None

        def _release_slot(slot: int):
            try:
                if slot in slot_to_tid:
                    del slot_to_tid[slot]
            except Exception:
                pass

        def _cleanup_pend(st: dict):
            for p in st.get("pend_files", []):
                try:
                    if p and os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass
            st["pend_files"] = []

        def _candidate_score(best_conf: float, cnt: int, sharp: float) -> float:
            # score đơn giản nhưng hiệu quả: conf + bonus vote + bonus sharp
            return float(best_conf) + 0.05 * float(cnt) + (0.03 if sharp > 160 else 0.0)

        def _add_vote(st: dict, plate_raw: str, conf: float, crop_bgr: np.ndarray, sharp: float, source: str):
            plate, kind = validate_and_canonicalize_plate(plate_raw)
            logging.debug(
                f"[{self.camera_name}] T{st['tid']} OCR({source}) raw='{plate_raw}' conf={conf:.2f} sharp={sharp:.1f} -> plate='{plate}' kind='{kind}'"
            )
            if not plate:
                return

            v = st["votes"].get(plate)
            if v is None:
                v = {"cnt": 0, "best_conf": 0.0, "best_img": None, "best_sharp": 0.0, "kind": kind or ""}
                st["votes"][plate] = v

            v["cnt"] += 1
            if (conf > v["best_conf"]) or (abs(conf - v["best_conf"]) < 1e-6 and sharp > v["best_sharp"]):
                v["best_conf"] = float(conf)
                v["best_sharp"] = float(sharp)
                v["best_img"] = crop_bgr.copy()
                v["kind"] = kind or v.get("kind", "")

            # update best_plate of track
            cur_best_plate = st.get("best_plate", "")
            cur_best_cnt = int(st["votes"].get(cur_best_plate, {}).get("cnt", 0)) if cur_best_plate else 0
            cur_best_conf = float(st.get("best_conf", 0.0))
            cur_best_sharp = float(st.get("best_sharp", 0.0))

            new_score = _candidate_score(v["best_conf"], v["cnt"], v["best_sharp"])
            cur_score = _candidate_score(cur_best_conf, cur_best_cnt, cur_best_sharp)

            if (plate == cur_best_plate and new_score > cur_score) or (new_score > cur_score):
                st["best_plate"] = plate
                st["best_kind"] = v.get("kind", "") or ""
                st["best_conf"] = float(v["best_conf"])
                st["best_img"] = v["best_img"]
                st["best_sharp"] = float(v["best_sharp"])
                st["best_cnt"] = int(v["cnt"])
                st["best_score"] = float(new_score)

        def _write_snapshot(st: dict, plate: str) -> str:
            img = st.get("best_img")
            if img is None:
                return ""

            ts = datetime.now()
            day_dir = os.path.join(self.snapshot_dir, ts.strftime("%Y-%m-%d"))
            os.makedirs(day_dir, exist_ok=True)
            snap_path = os.path.join(day_dir, f"{plate}_{self.camera_name}_{int(time.time()*1000)}.jpg")
            try:
                cv2.imwrite(snap_path, img)
                return snap_path
            except Exception:
                return ""

        def _emit_upsert(st: dict, final: bool, reason: str):
            plate = st.get("best_plate", "") or ""
            if not plate:
                return

            # write best snapshot on demand (update overwrite by new file, delete old one)
            new_snap = _write_snapshot(st, plate)
            if new_snap:
                old = st.get("last_emit_snap", "")
                if old and old != new_snap and os.path.exists(old):
                    try:
                        os.remove(old)
                    except Exception:
                        pass
                st["last_emit_snap"] = new_snap

            payload = {
                "plate": plate,
                "kind": st.get("best_kind", "") or "",
                "role": self.role,
                "camera_name": self.camera_name,
                "ts": datetime.now(),
                "det_conf": float(st.get("last_det_conf", 0.0) or 0.0),
                "ocr_conf": float(st.get("best_conf", 0.0) or 0.0),
                "sharpness": float(st.get("best_sharp", 0.0) or 0.0),
                "snapshot_path": new_snap or st.get("last_emit_snap", "") or "",
                "focus_id": st.get("focus_id", "") or "",
                "track_id": int(st.get("tid", 0) or 0),
                "vote_cnt": int(st.get("best_cnt", 0) or 0),
                "best_score": float(st.get("best_score", 0.0) or 0.0),
                "final": bool(final),
                "reason": str(reason),
            }

            # emit
            try:
                self.plate_event.emit(payload)
            except Exception:
                pass

            st["event_sent"] = True
            st["last_sent_score"] = float(st.get("best_score", 0.0) or 0.0)
            st["last_sent_ts"] = time.time()

            logging.info(
                f"[UPSERT] {self.camera_name} T{st['tid']} plate={plate} kind={payload['kind']} conf={payload['ocr_conf']:.2f} "
                f"votes={payload['vote_cnt']} sharp={payload['sharpness']:.1f} final={payload['final']} reason={reason}"
            )

        def _maybe_commit_or_update(st: dict):
            if not st.get("best_plate"):
                return

            now = time.time()
            age = now - float(st.get("start", now))

            best_cnt = int(st.get("best_cnt", 0) or 0)
            best_conf = float(st.get("best_conf", 0.0) or 0.0)

            # insert lần đầu
            if not st.get("event_sent", False):
                if age < min_track_age_to_commit:
                    return
                if (best_cnt >= vote_need and best_conf >= min_final_conf) or (best_conf >= high_conf_one_shot):
                    _emit_upsert(st, final=False, reason="stable")
                return

            # update nếu score tốt hơn đủ nhiều
            last_score = float(st.get("last_sent_score", 0.0) or 0.0)
            cur_score = float(st.get("best_score", 0.0) or 0.0)

            if cur_score > last_score + update_score_delta:
                if now - float(st.get("last_sent_ts", 0.0) or 0.0) >= update_min_interval:
                    _emit_upsert(st, final=False, reason="improve")

        def _finalize_track(tid: int, reason: str):
            st = tracks.get(tid)
            if not st:
                return

            # nếu chưa emit lần nào nhưng cuối track đủ điều kiện -> emit 1 lần FINAL
            if not st.get("event_sent", False):
                best_cnt = int(st.get("best_cnt", 0) or 0)
                best_conf = float(st.get("best_conf", 0.0) or 0.0)
                if st.get("best_plate") and ((best_cnt >= vote_need and best_conf >= min_final_conf) or (best_conf >= high_conf_one_shot)):
                    _emit_upsert(st, final=True, reason=reason)
                else:
                    logging.info(
                        f"[DROP] {self.camera_name} T{tid} finalize(reason={reason}) best='{st.get('best_plate','')}' "
                        f"conf={best_conf:.2f} votes={best_cnt}"
                    )
            else:
                # đã emit -> gửi FINAL để main.py cleanup map
                _emit_upsert(st, final=True, reason=reason)

            # cleanup
            _cleanup_pend(st)
            fid = st.get("focus_id", "")
            if fid in focus_to_tid:
                try:
                    del focus_to_tid[fid]
                except Exception:
                    pass

            slot = int(st.get("slot", -1))
            if slot >= 0:
                _release_slot(slot)

            try:
                del tracks[tid]
            except Exception:
                pass

        # main loop
        while self._running:
            with self._frame_lock:
                frame = None if self._latest_frame is None else self._latest_frame.copy()
            if frame is None:
                time.sleep(0.01)
                continue

            frame_idx += 1
            now_ts = time.time()

            # 1) poll enhance results
            try:
                if self._enhancer:
                    for (focus_id, cam, zone, plate_raw, conf, enh_img, sharp) in self._enhancer.poll():
                        if cam != self.camera_name:
                            continue
                        tid = focus_to_tid.get(str(focus_id))
                        if not tid:
                            continue
                        st = tracks.get(int(tid))
                        if not st:
                            continue
                        _add_vote(st, plate_raw, float(conf), enh_img, float(sharp), source="enh")
                        _maybe_commit_or_update(st)
            except Exception:
                logging.exception(f"[{self.camera_name}] enhance poll error")

            # 2) motion roi
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            motion_detected = False
            if prev_gray is not None:
                diff = cv2.absdiff(prev_gray, gray)
                _, th = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                th = cv2.dilate(th, None, iterations=2)

                cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                x1m, y1m, x2m, y2m = 10**9, 10**9, 0, 0
                motion_area_sum = 0.0

                for c in cnts:
                    area = cv2.contourArea(c)
                    if area < 1200:
                        continue
                    x, y, w1, h1 = cv2.boundingRect(c)
                    motion_area_sum += float(area)
                    x1m = min(x1m, x); y1m = min(y1m, y)
                    x2m = max(x2m, x + w1); y2m = max(y2m, y + h1)

                if motion_area_sum > 2500 and x2m > x1m and y2m > y1m:
                    motion_detected = True
                    pad = 40
                    h0, w0 = frame.shape[:2]
                    x1m = max(0, x1m - pad); y1m = max(0, y1m - pad)
                    x2m = min(w0, x2m + pad); y2m = min(h0, y2m + pad)
                    motion_roi = (x1m, y1m, x2m, y2m)
                    motion_hold_until = now_ts + 0.7

            prev_gray = gray
            if motion_roi is not None and now_ts > motion_hold_until:
                motion_roi = None

            do_detect = motion_detected or (frame_idx % self.detect_every_n == 0)
            if not do_detect:
                time.sleep(0.01)
                continue

            # 3) detect plates
            # If motion_roi is active, we still force a full-frame detection periodically.
            use_roi = motion_roi
            if use_roi is not None and fullframe_sec > 0:
                if (now_ts - float(last_fullframe_detect_ts)) >= float(fullframe_sec):
                    use_roi = None
            try:
                dets = self.alpr.detect_multi(frame, max_plates=max_plate_dets, roi=use_roi)
                if use_roi is None:
                    last_fullframe_detect_ts = now_ts
            except Exception as e:
                logging.exception(f"[{self.camera_name}] detect_multi error: {type(e).__name__}: {e}")
                time.sleep(0.05)
                continue

            # remove duplicated detections (YOLO sometimes returns 2 close boxes)
            if dets:
                dets2 = []
                for d in sorted(dets, key=lambda x: float(x.get("det_conf", 0.0)), reverse=True):
                    bb = d.get("bbox")
                    if not bb:
                        continue
                    dup = False
                    for k in dets2:
                        if _iou(bb, k["bbox"]) >= duplicate_det_iou:
                            dup = True
                            break
                    if not dup:
                        dets2.append(d)
                dets = dets2[:max_plate_dets]

            # 4) match dets -> tracks
            matched_tids = set()
            matched_det_idx = set()
            pairs = []
            for tid, st in tracks.items():
                bb_track = st.get("bbox")
                if not bb_track:
                    continue
                assoc_bb = st.get("assoc_bbox") or st.get("bbox")
                tcx, tcy = _bbox_center(bb_track)
                max_dist = _assoc_max_dist(bb_track)
                for di, d in enumerate(dets or []):
                    bb = d.get("bbox")
                    if not bb:
                        continue
                    iou_val = _iou(bb_track, bb)

                    # distance + inside-gate (small assoc bbox)
                    dcx, dcy = _bbox_center(bb)
                    dist = math.hypot(float(dcx - tcx), float(dcy - tcy))
                    inside = 0
                    if assoc_bb is not None and dist <= max_dist:
                        if _point_in_bb((dcx, dcy), assoc_bb):
                            inside = 1

                    score = iou_val
                    # if the plate bbox jitters (low IOU) but still belongs to the same track,
                    # allow a weaker association score. Keep it strict to avoid merging tracks.
                    if inside and score < iou_match_thres:
                        score = max(score, inside_match_score)

                    pairs.append((score, iou_val, inside, tid, di))

            pairs.sort(key=lambda x: x[0], reverse=True)

            matches: List[Tuple[int, int]] = []
            for score, iou_val, inside, tid, di in pairs:
                # stop early when remaining scores are too weak
                if (iou_val < iou_match_thres) and (score < inside_match_score):
                    break
                if tid in matched_tids or di in matched_det_idx:
                    continue

                if iou_val >= iou_match_thres:
                    pass
                else:
                    if not inside:
                        continue
                    # also require track not already considered lost
                    st = tracks.get(tid)
                    if not st:
                        continue
                    last_seen = float(st.get("last_seen", now_ts) or now_ts)
                    if now_ts - last_seen > track_lost_sec:
                        continue

                matches.append((tid, di))
                matched_tids.add(tid)
                matched_det_idx.add(di)

            # unmatched dets will create new tracks (if slot available)
            unmatched_det = [i for i in range(len(dets or [])) if i not in matched_det_idx]

            # update matched tracks
            for tid, di in matches:
                st = tracks.get(tid)
                d = dets[di]
                bb = d.get("bbox")
                crop = d.get("crop")
                if st is None or bb is None or crop is None:
                    continue

                st["bbox"] = tuple(map(int, bb))
                st["assoc_bbox"] = _expand_assoc_bbox(st["bbox"], frame.shape)
                st["obj_bbox"] = _expand_obj_bbox(st["bbox"], frame.shape, st.get("best_kind", ""))
                st["last_seen"] = now_ts
                st["lost_since"] = 0.0
                st["last_det_conf"] = float(d.get("det_conf", 0.0) or 0.0)

                # UI focus image (only for tracks that own a UI slot)
                if int(st.get("slot", -1)) >= 0:
                    try:
                        self.focus_ready.emit(self.camera_name, int(st["slot"]), bgr_to_qimage(crop.copy()))
                    except Exception:
                        pass

                sharp = _laplacian_var(crop)

                # pending crops (optional)
                if len(st["pend_files"]) < max_pend:
                    ts = datetime.now()
                    day_dir = os.path.join(self.snapshot_dir, ts.strftime("%Y-%m-%d"))
                    os.makedirs(day_dir, exist_ok=True)
                    pend_path = os.path.join(day_dir, f"PEND_{st['focus_id']}_{len(st['pend_files'])}.jpg")
                    try:
                        cv2.imwrite(pend_path, crop)
                        st["pend_files"].append(pend_path)
                    except Exception:
                        pass

                # OCR fast
                try:
                    plate_raw, ocr_conf = self.alpr.read_plate_text(crop, fast=True)
                except TypeError:
                    plate_raw, ocr_conf = self.alpr.read_plate_text(crop)

                _add_vote(st, plate_raw, float(ocr_conf), crop, float(sharp), source="raw")

                # enhance slow path if needed (throttled)
                if self._enhancer and (float(ocr_conf) < enhance_trigger_conf or sharp < sharp_trigger):
                    if now_ts - float(st.get("last_enhance_submit", 0.0) or 0.0) >= enhance_min_interval:
                        st["last_enhance_submit"] = now_ts
                        if int(st.get("slot", -1)) >= 0:
                            self._enhancer.submit(st["focus_id"], self.camera_name, int(st["slot"]), crop)

                _maybe_commit_or_update(st)

            # create new tracks for unmatched detections
            for di in unmatched_det:
                d = dets[di]
                bb = d.get("bbox")
                crop = d.get("crop")
                if bb is None or crop is None:
                    continue

                if len(tracks) >= max_tracks:
                    continue

                slot = _alloc_slot()
                if slot is None:
                    slot = -1

                tid = next_tid
                next_tid += 1

                focus_id = f"{self.camera_name}_T{tid}_{int(now_ts*1000)}"

                st = {
                    "tid": tid,
                    "slot": slot,
                    "focus_id": focus_id,
                    "bbox": tuple(map(int, bb)),
                    "assoc_bbox": _expand_assoc_bbox(tuple(map(int, bb)), frame.shape),
                    "obj_bbox": _expand_obj_bbox(tuple(map(int, bb)), frame.shape, ""),
                    "start": now_ts,
                    "last_seen": now_ts,
                    "lost_since": 0.0,
                    "votes": {},
                    "best_plate": "",
                    "best_kind": "",
                    "best_conf": 0.0,
                    "best_img": None,
                    "best_sharp": 0.0,
                    "best_cnt": 0,
                    "best_score": 0.0,
                    "event_sent": False,
                    "last_sent_score": 0.0,
                    "last_sent_ts": 0.0,
                    "last_emit_snap": "",
                    "last_det_conf": float(d.get("det_conf", 0.0) or 0.0),
                    "pend_files": [],
                    "last_enhance_submit": 0.0,
                }

                tracks[tid] = st
                focus_to_tid[focus_id] = tid
                if slot >= 0:
                    slot_to_tid[int(slot)] = tid

                logging.info(f"[{self.camera_name}] START track T{tid} slot={slot if int(slot) >= 0 else '-'} focus_id={focus_id}")

                # show focus immediately (only if this track owns a UI slot)
                if int(slot) >= 0:
                    try:
                        self.focus_ready.emit(self.camera_name, int(slot), bgr_to_qimage(crop.copy()))
                    except Exception:
                        pass

                sharp = _laplacian_var(crop)
                try:
                    plate_raw, ocr_conf = self.alpr.read_plate_text(crop, fast=True)
                except TypeError:
                    plate_raw, ocr_conf = self.alpr.read_plate_text(crop)

                _add_vote(st, plate_raw, float(ocr_conf), crop, float(sharp), source="raw")

                if self._enhancer and int(slot) >= 0 and (float(ocr_conf) < enhance_trigger_conf or sharp < sharp_trigger):
                    st["last_enhance_submit"] = now_ts
                    self._enhancer.submit(st["focus_id"], self.camera_name, int(slot), crop)

                _maybe_commit_or_update(st)

            # finalize tracks that are lost
            for tid, st in list(tracks.items()):
                if tid in matched_tids:
                    continue

                # not matched this round
                last_seen = float(st.get("last_seen", now_ts))
                if now_ts - last_seen >= track_lost_sec:
                    _finalize_track(tid, reason="lost")
            # update overlay for UI
            overlays = []
            for tid, st in tracks.items():
                bb = st.get("obj_bbox") or st.get("bbox")
                if not bb:
                    continue
                plate = st.get("best_plate", "") or ""
                kind = st.get("best_kind", "") or ""
                conf = float(st.get("best_conf", 0.0) or 0.0)
                sent = bool(st.get("event_sent", False))

                if plate:
                    label = f"T{tid} {plate} {conf:.2f}" + (f" {kind}" if kind else "")
                else:
                    label = f"T{tid}" + (f" {kind}" if kind else "")

                color = (0, 255, 0) if not sent else (0, 255, 255)  # green=tracking, yellow=logged
                overlays.append({"bbox": bb, "label": label, "color": color})

            # motion roi overlay (optional)
            if draw_motion_roi and motion_roi is not None:
                overlays.append({"bbox": motion_roi, "label": "MOTION", "color": (255, 0, 255)})

            with self._overlay_lock:

                self._overlay_boxes = overlays

            time.sleep(0.01)
