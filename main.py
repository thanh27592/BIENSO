import json
import os
import sys
import re
import time
import logging
from logging.handlers import RotatingFileHandler
from PySide6.QtWidgets import QApplication

from alpr import ALPR, normalize_plate, validate_and_canonicalize_plate
from db import DB
from camera_worker import CameraWorker
from ui import MainUI

from google_oauth_banhang_style import get_drive_service, TOKEN_PATH

def setup_logging(log_path: str = "anpr_debug.log"):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s")

    # clear handlers to avoid duplicates
    for h in list(logger.handlers):
        logger.removeHandler(h)

    # File (rotate)
    fh = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    # Console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)


def load_config(path: str = "config.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_valid_vn_plate(p: str) -> bool:
    """
    p đã là chuỗi normalize kiểu: 50A155555, 50AA55555, 50H55555 ...
    """
    if not p:
        return False

    patterns = [
        r"^\d{2}[A-Z]\d\d{5}$",     # 9: moto (59V168055...)
        r"^\d{2}[A-Z]{2}\d{5}$",    # 9: moto 2 letters (50AA58888)
        r"^\d{2}[A-Z]\d{5}$",       # 8: car phổ biến (68A56748)
        r"^\d{2}[A-Z]\d{4}$",       # 7: car cũ (tùy)
    ]
    return any(re.match(pt, p) for pt in patterns)


def main():
    cfg = load_config()
    setup_logging(cfg.get("log_path", "anpr_debug.log"))

        # ================== GOOGLE OAUTH (BANHANG STYLE) ==================
    # Mục tiêu: Khi BIENSO chạy sẽ mở trình duyệt yêu cầu đăng nhập Google (nếu chưa có token)
    # và lưu token.json vào: %LOCALAPPDATA%\Token\token.json (giống BANHANG).
    drive_service = None
    try:
        drive_service = get_drive_service()  # sẽ tự mở Chrome incognito/guest nếu cần đăng nhập
        logging.info(f"[GOOGLE] Drive OAuth OK. token saved at: {TOKEN_PATH}")
    except Exception:
        logging.exception("[GOOGLE] Drive OAuth failed. App will continue without Drive.")
        drive_service = None


    excel_path = cfg.get("excel_path", "anpr_export.xlsx")
    ignore_set = set(normalize_plate(x) for x in cfg.get("ignore_plates", []))

    model_path = cfg["model_path"]
    if not os.path.exists(model_path):
        print(f"[ERROR] Missing model file: {model_path}")
        print("Download it first (see step 2).")
        sys.exit(1)

    snapshot_dir = cfg.get("snapshot_dir", "snapshots")
    os.makedirs(snapshot_dir, exist_ok=True)

    db = DB(cfg.get("db_path", "anpr.db"))
    alpr = ALPR(model_path=model_path, conf_thres=float(cfg.get("conf_thres", 0.35)))

    app = QApplication(sys.argv)
    ui = MainUI(db=db, snapshot_dir=snapshot_dir)
    # Cho phép UI (hoặc module khác) dùng Drive service nếu cần upload
    ui.drive_service = drive_service
    ui.show()

    workers = []
    debounce_sec = int(cfg.get("debounce_sec", 12))

    # focus_id -> {event_id, session_id, created_ts, last_ts}
    upsert_map = {}
    UPSERT_TTL_SEC = 10 * 60  # 10 phút

    def _prune_upserts():
        now = time.time()
        dead = [k for k, v in upsert_map.items() if now - float(v.get("last_ts", v.get("created_ts", now))) > UPSERT_TTL_SEC]
        for k in dead:
            try:
                del upsert_map[k]
            except Exception:
                pass

    def _on_event(payload, _db=db, _ui=ui, _cfg=cfg, _ignore=ignore_set):
        """
        payload do camera_worker emit:
        - plate (canonical)
        - kind, det_conf, ocr_conf, sharpness
        - snapshot_path
        - focus_id (định danh "1 đối tượng" / track)
        - final (True khi track kết thúc)
        """
        try:
            _prune_upserts()

            plate = payload.get("plate", "") or ""
            role = payload.get("role", "") or ""
            camname = payload.get("camera_name", "") or ""
            ts = payload.get("ts")
            det_conf = float(payload.get("det_conf", 0.0) or 0.0)
            ocr_conf = float(payload.get("ocr_conf", payload.get("conf", 0.0) or 0.0) or 0.0)
            sharpness = float(payload.get("sharpness", 0.0) or 0.0)
            snap = payload.get("snapshot_path", "") or ""
            kind = payload.get("kind", "") or ""
            focus_id = payload.get("focus_id", "") or ""
            final = bool(payload.get("final", False))

            # 0) Defensive: nếu thiếu ts
            if ts is None:
                logging.warning(f"[DROP] Missing ts from payload: {payload}")
                return

            # 1) Validate & canonicalize lại lần nữa cho chắc
            plate2, kind2 = validate_and_canonicalize_plate(plate)
            if not plate2:
                logging.debug(f"[DROP] plate not canonical: raw={plate}")
                if snap and os.path.exists(snap):
                    try:
                        os.remove(snap)
                    except Exception:
                        pass
                return
            plate = plate2
            if not kind:
                kind = kind2

            # 2) ignore list (config)
            if plate in _ignore:
                logging.info(f"[IGNORE] {plate} in ignore_plates (config)")
                if snap and os.path.exists(snap):
                    try:
                        os.remove(snap)
                    except Exception:
                        pass
                # cleanup map if final
                if final and focus_id and focus_id in upsert_map:
                    del upsert_map[focus_id]
                return

            # 3) strict VN rule
            if _cfg.get("strict_vn_plate", True):
                if not is_valid_vn_plate(plate):
                    logging.info(f"[DROP] Not valid VN plate: {plate}")
                    if snap and os.path.exists(snap):
                        try:
                            os.remove(snap)
                        except Exception:
                            pass
                    if final and focus_id and focus_id in upsert_map:
                        del upsert_map[focus_id]
                    return

            # 4) optional filter by kind
            allowed_kinds = _cfg.get("allowed_kinds", None)
            if isinstance(allowed_kinds, list) and allowed_kinds:
                if kind and kind not in allowed_kinds:
                    logging.info(f"[DROP] kind={kind} not in allowed_kinds={allowed_kinds} plate={plate}")
                    if snap and os.path.exists(snap):
                        try:
                            os.remove(snap)
                        except Exception:
                            pass
                    if final and focus_id and focus_id in upsert_map:
                        del upsert_map[focus_id]
                    return

            # 5) UPSERT theo focus_id (mỗi đối tượng/track chỉ 1 log)
            plate_key = _db.make_plate_key(plate)
            min_out_delay_sec = _ui.get_min_out_delay_sec()

            if focus_id and focus_id in upsert_map:
                meta = upsert_map[focus_id]
                event_id = int(meta["event_id"])
                session_id = int(meta["session_id"]) if meta.get("session_id") else 0

                _db.update_event(
                    event_id,
                    plate=plate,
                    snapshot_path=(snap if snap else None),
                    kind=kind,
                    det_conf=det_conf,
                    ocr_conf=ocr_conf,
                    sharpness=sharpness,
                )
                if session_id > 0:
                    _db.update_session_plate(session_id, plate_final=plate, plate_key=plate_key)

                meta["last_ts"] = time.time()
                logging.info(f"[UPDATE] focus_id={focus_id} event_id={event_id} session_id={session_id} plate={plate} conf={ocr_conf:.2f} sharp={sharpness:.1f}")

                if final:
                    del upsert_map[focus_id]

            else:
                # INSERT event
                event_id = _db.insert_event(
                    plate=plate,
                    role=role,
                    camera_name=camname,
                    ts=ts,
                    conf=ocr_conf,
                    snapshot_path=snap,
                    focus_id=focus_id,
                    kind=kind,
                    det_conf=det_conf,
                    ocr_conf=ocr_conf,
                    sharpness=sharpness,
                )

                action, sid = _db.process_warehouse_event(
                    plate_final=plate,
                    plate_key=plate_key,
                    event_id=event_id,
                    ts=ts,
                    min_out_delay_sec=min_out_delay_sec,
                )
                logging.info(f"[SESSION] action={action} sid={sid} plate={plate} key={plate_key} min_out={min_out_delay_sec}s (event_id={event_id})")

                if focus_id:
                    upsert_map[focus_id] = {
                        "event_id": event_id,
                        "session_id": sid,
                        "created_ts": time.time(),
                        "last_ts": time.time(),
                    }

                # if final -> immediately clear
                if final and focus_id and focus_id in upsert_map:
                    del upsert_map[focus_id]

            # 6) UI notify
            _ui.on_plate_event(payload)

        except Exception:
            logging.exception(f"[ERROR] _on_event crashed. payload={payload}")
            return

    for cam in cfg["cameras"]:
        w = CameraWorker(
            camera_name=cam["name"],
            role=cam["role"],
            rtsp=cam["rtsp"],
            alpr=alpr,
            snapshot_dir=snapshot_dir,
            debounce_sec=debounce_sec,
            detect_every_n=int(cam.get("detect_every_n", 2) or 2),
        )
        w.frame_ready.connect(ui.update_frame)
        w.status.connect(ui.set_status)
        w.focus_ready.connect(ui.update_focus)
        w.plate_event.connect(_on_event)
        workers.append(w)

    for w in workers:
        w.start()

    code = app.exec()

    for w in workers:
        w.stop()
        w.wait(1000)

    sys.exit(code)


if __name__ == "__main__":
    main()
