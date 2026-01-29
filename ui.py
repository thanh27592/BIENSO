import os
import json
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap, QImage
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QLabel, QHBoxLayout, QVBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QHeaderView, QMessageBox, QSpinBox, QInputDialog,
    QDateEdit, QFileDialog
)

from datetime import datetime
from alpr import validate_and_canonicalize_plate

def bgr_to_qimage(bgr):
    """
    Convert OpenCV BGR ndarray -> QImage (RGB888).
    Trả về QImage đã copy để không bị lỗi lifetime bộ nhớ.
    FIX: đảm bảo buffer C-contiguous (stride dương), tránh BufferError.
    """
    if bgr is None:
        return QImage()

    # đảm bảo bgr liên tục trong memory
    if not bgr.flags["C_CONTIGUOUS"]:
        bgr = np.ascontiguousarray(bgr)

    # BGR -> RGB
    if bgr.ndim == 3 and bgr.shape[2] == 3:
        # NOTE: bgr[:, :, ::-1] tạo view stride âm => phải ép contiguous lại
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])

        h, w = rgb.shape[:2]
        bytes_per_line = 3 * w  # RGB888

        return QImage(
            rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
        ).copy()

    # grayscale fallback
    if bgr.ndim == 2:
        g = bgr
        if not g.flags["C_CONTIGUOUS"]:
            g = np.ascontiguousarray(g)

        h, w = g.shape[:2]
        bytes_per_line = w
        return QImage(
            g.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8
        ).copy()

    return QImage()

class MainUI(QWidget):
    def __init__(self, db, snapshot_dir):
        super().__init__()
        self.db = db
        self.snapshot_dir = snapshot_dir

        self.setWindowTitle("Hikvision Gate ANPR (MVP)")
        self.resize(1300, 800)

        # video panels
        self.lbl_in = QLabel("IN CAMERA")
        self.lbl_out = QLabel("OUT CAMERA")
        for lbl in (self.lbl_in, self.lbl_out):
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setMinimumSize(560, 315)
            lbl.setStyleSheet("background:#111; color:#ddd; border:1px solid #333;")

        # ui.py - trong __init__ sau khi tạo lbl_in/lbl_out
        def _mk_focus_label(title: str):
            lb = QLabel(title)
            lb.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lb.setMinimumSize(270, 160)
            lb.setStyleSheet("background:#0b0b0b; color:#aaa; border:1px solid #333;")
            return lb

        self.focus_in_1 = _mk_focus_label("IN PLATE #1")
        self.focus_in_2 = _mk_focus_label("IN PLATE #2")
        self.focus_out_1 = _mk_focus_label("OUT PLATE #1")
        self.focus_out_2 = _mk_focus_label("OUT PLATE #2")

        self.status_in = QLabel("IN: -")
        self.status_out = QLabel("OUT: -")

        # datalog (1 bảng)
        self.tbl_log = QTableWidget(0, 5)
        self.tbl_log.setHorizontalHeaderLabels(["Time IN", "Time OUT", "Plate", "Duration(HH:MM)", "Snapshot"])
        self.tbl_log.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.tbl_log.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        self.btn_refresh = QPushButton("Refresh DB")
        self.btn_refresh.clicked.connect(self.refresh_tables)

        # ---- Export Excel theo ngày ----
        self.date_export = QDateEdit()
        self.date_export.setCalendarPopup(True)
        self.date_export.setDisplayFormat("yyyy-MM-dd")
        self.date_export.setDate(datetime.now().date())

        self.btn_export = QPushButton("Xuất Excel")
        self.btn_export.clicked.connect(self.export_excel_for_day)

        # ---- Danh sách xe tháng (ignore list) ----
        self.btn_month_list = QPushButton("DS xe tháng")
        self.btn_month_list.clicked.connect(self.edit_month_list)
        self._ignore_file = "ignore_month.json"
        self._ignore_full = set()
        self._ignore_key_to_kinds = {}
        self._load_ignore_list()
        # min delay OUT (minutes) - default 5
        self.spin_min_out = QSpinBox()
        self.spin_min_out.setRange(0, 120)
        self.spin_min_out.setValue(5)
        self.spin_min_out.setSuffix(" min")

        self.btn_set_min_out = QPushButton("Set OUT min-delay")
        self.btn_set_min_out.clicked.connect(self._set_min_out_delay_with_password)

        # layout
        top = QHBoxLayout()
        left = QVBoxLayout()
        right = QVBoxLayout()

        # left layout
        left.addWidget(self.lbl_in)

        row_in = QHBoxLayout()
        row_in.addWidget(self.focus_in_1)
        row_in.addWidget(self.focus_in_2)
        left.addLayout(row_in)

        left.addWidget(self.status_in)

        # right layout
        right.addWidget(self.lbl_out)

        row_out = QHBoxLayout()
        row_out.addWidget(self.focus_out_1)
        row_out.addWidget(self.focus_out_2)
        right.addLayout(row_out)

        right.addWidget(self.status_out)

        top.addLayout(left)
        top.addLayout(right)

        main = QVBoxLayout()
        main.addLayout(top)
        row_cfg = QHBoxLayout()
        row_cfg.addWidget(self.btn_refresh)
        row_cfg.addWidget(self.date_export)
        row_cfg.addWidget(self.btn_export)
        row_cfg.addWidget(self.btn_month_list)
        row_cfg.addWidget(QLabel("Min OUT delay:"))
        row_cfg.addWidget(self.spin_min_out)
        row_cfg.addWidget(self.btn_set_min_out)
        main.addLayout(row_cfg)
        main.addWidget(QLabel("Datalog (latest)"))
        main.addWidget(self.tbl_log)

        self.setLayout(main)

        # auto refresh
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_tables)
        self.timer.start(1500)

    def set_status(self, camera_name, msg):
        if "IN" in camera_name.upper():
            self.status_in.setText(f"{camera_name}: {msg}")
        else:
            self.status_out.setText(f"{camera_name}: {msg}")

    def update_frame(self, camera_name, qimg):
        pix = QPixmap.fromImage(qimg).scaled(560, 315, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        if "IN" in camera_name.upper():
            self.lbl_in.setPixmap(pix)
        else:
            self.lbl_out.setPixmap(pix)

    def update_focus(self, camera_name, idx, qimg):
        pix = QPixmap.fromImage(qimg).scaled(270, 160, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

        is_in = "IN" in camera_name.upper()
        if is_in:
            target = self.focus_in_1 if idx == 0 else self.focus_in_2
        else:
            target = self.focus_out_1 if idx == 0 else self.focus_out_2

        target.setPixmap(pix)

    def on_plate_event(self, payload):
        # popup nhẹ (tuỳ bạn, có thể tắt nếu khó chịu)
        plate = payload["plate"]
        role = payload["role"]
        cam = payload["camera_name"]
        self.setWindowTitle(f"Last: {role} {plate} ({cam})")

    def _fmt_hhmm(self, seconds):
        if seconds is None:
            return ""
        try:
            seconds = int(seconds)
        except Exception:
            return ""
        if seconds < 0:
            seconds = 0
        m = seconds // 60
        hh = m // 60
        mm = m % 60
        return f"{hh:02d}:{mm:02d}"

    def refresh_tables(self):
        try:
            rows = self.db.get_recent_datalog(120)
        except Exception as e:
            QMessageBox.warning(self, "DB Error", str(e))
            return

        self.tbl_log.setRowCount(0)
        for r, s in enumerate(rows):
            self.tbl_log.insertRow(r)

            time_in = s.get("time_in", "") or ""
            time_out = s.get("time_out", "") or ""
            plate = s.get("plate", "") or ""
            dur = self._fmt_hhmm(s.get("duration_sec", None))

            self.tbl_log.setItem(r, 0, QTableWidgetItem(str(time_in)))
            self.tbl_log.setItem(r, 1, QTableWidgetItem(str(time_out)))
            self.tbl_log.setItem(r, 2, QTableWidgetItem(str(plate)))
            self.tbl_log.setItem(r, 3, QTableWidgetItem(str(dur)))

            # Snapshot: ưu tiên OUT nếu có, nếu chưa thì lấy IN
            snap = s.get("snap_out") or s.get("snap_in") or ""

            img_label = QLabel()
            img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

            if snap and os.path.exists(snap):
                pix = QPixmap(snap)
                if not pix.isNull():
                    pix = pix.scaled(
                        160, 90,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation
                    )
                    img_label.setPixmap(pix)
                img_label.setToolTip(snap)
            else:
                img_label.setText(os.path.basename(snap) if snap else "")
                img_label.setToolTip(snap)

            self.tbl_log.setCellWidget(r, 4, img_label)

    def _set_min_out_delay_with_password(self):
        pwd, ok = QInputDialog.getText(self, "Password", "Enter password to change:", echo=QInputDialog.EchoMode.Password)
        if not ok:
            return
        if pwd.strip() != "27592":
            QMessageBox.warning(self, "Denied", "Wrong password")
            return
        QMessageBox.information(self, "OK", f"Min OUT delay set to {self.spin_min_out.value()} minutes")

    def get_min_out_delay_sec(self) -> int:
        return int(self.spin_min_out.value()) * 60
    
    # -------------------------
    # IGNORE LIST (DS xe tháng)
    # -------------------------
    def _plate_key_fallback(self, plate: str) -> str:
        """Fallback nếu db.make_plate_key không có (lấy 4 số cuối)."""
        digits = "".join(ch for ch in plate if ch.isdigit())
        return digits[-4:] if len(digits) >= 4 else ""

    def _rebuild_ignore_index(self):
        """Tạo index: full plate + (plate_key -> kinds) để match theo rule 2.c."""
        self._ignore_full = set()
        self._ignore_key_to_kinds = {}

        for p in getattr(self, "_ignore_list", []):
            plate, kind = validate_and_canonicalize_plate(p)
            if not plate:
                continue

            self._ignore_full.add(plate)

            # plate_key để so 4 số cuối
            try:
                k = self.db.make_plate_key(plate)
            except Exception:
                k = self._plate_key_fallback(plate)

            if not k:
                continue

            if k not in self._ignore_key_to_kinds:
                self._ignore_key_to_kinds[k] = set()

            # lưu kind để giảm nhầm (CAR/MOTO)
            if kind:
                self._ignore_key_to_kinds[k].add(kind)
            else:
                self._ignore_key_to_kinds[k].add("")

    def _load_ignore_list(self):
        """Load ignore_month.json -> self._ignore_list."""
        self._ignore_list = []
        try:
            if os.path.exists(self._ignore_file):
                with open(self._ignore_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                plates = data.get("plates", [])
                if isinstance(plates, list):
                    self._ignore_list = [str(x).strip() for x in plates if str(x).strip()]
        except Exception:
            self._ignore_list = []

        self._rebuild_ignore_index()

    def _save_ignore_list(self):
        """Save self._ignore_list -> ignore_month.json."""
        try:
            with open(self._ignore_file, "w", encoding="utf-8") as f:
                json.dump({"plates": self._ignore_list}, f, ensure_ascii=False, indent=2)
        except Exception as e:
            QMessageBox.warning(self, "Save error", f"Không lưu được {self._ignore_file}\n{e}")

    def edit_month_list(self):
        """UI nhập DS xe tháng (multi-line)."""
        current = "\n".join(getattr(self, "_ignore_list", []))
        text, ok = QInputDialog.getMultiLineText(
            self,
            "DS xe tháng",
            "Nhập danh sách biển số (mỗi dòng 1 biển). Các biển này sẽ KHÔNG được ghi IN/OUT:",
            current
        )
        if not ok:
            return

        raw_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        cleaned = []
        invalid = []

        for ln in raw_lines:
            plate, kind = validate_and_canonicalize_plate(ln)
            if not plate:
                invalid.append(ln)
                continue
            cleaned.append(plate)

        # unique giữ thứ tự
        seen = set()
        uniq = []
        for p in cleaned:
            if p not in seen:
                seen.add(p)
                uniq.append(p)

        self._ignore_list = uniq
        self._rebuild_ignore_index()
        self._save_ignore_list()

        msg = f"Đã lưu {len(self._ignore_list)} biển số vào DS xe tháng."
        if invalid:
            msg += f"\n\nKhông hợp lệ ({len(invalid)}):\n" + "\n".join(invalid[:20])
            if len(invalid) > 20:
                msg += "\n..."

        QMessageBox.information(self, "OK", msg)

    def is_ignored(self, plate: str, plate_key: str = "", kind: str = "") -> bool:
        """
        Match ignore theo yêu cầu 2.c:
        1) full plate trùng -> ignore
        2) nếu không có -> 4 số cuối trùng + xét kind (CAR/MOTO) nếu có
        """
        p, k = validate_and_canonicalize_plate(plate)
        if not p:
            return False

        if p in self._ignore_full:
            return True

        # plate_key ưu tiên dùng cái main.py truyền vào
        key = plate_key or ""
        if not key:
            try:
                key = self.db.make_plate_key(p)
            except Exception:
                key = self._plate_key_fallback(p)

        if not key:
            return False

        kinds = self._ignore_key_to_kinds.get(key)
        if not kinds:
            return False

        if not kind:
            # không biết loại xe -> nếu key trùng thì coi như ignore
            return True

        # nếu trong DS có đúng kind hoặc có item kind rỗng (chấp nhận mọi loại)
        return (kind in kinds) or ("" in kinds)


    # -------------------------
    # EXPORT EXCEL THEO NGÀY
    # -------------------------
    def export_excel_for_day(self):
        """Chọn ngày trên UI và export Excel."""
        day = self.date_export.date().toString("yyyy-MM-dd")

        # file save dialog
        default_name = f"anpr_{day}.xlsx"
        out_path, _ = QFileDialog.getSaveFileName(
            self,
            "Xuất Excel",
            default_name,
            "Excel (*.xlsx)"
        )
        if not out_path:
            return

        # gọi DB export
        try:
            self.db.export_day_to_excel(day, out_path)
        except Exception as e:
            QMessageBox.warning(self, "Export failed", f"Không xuất được Excel:\n{e}")
            return

        QMessageBox.information(self, "OK", f"Đã xuất Excel:\n{out_path}")