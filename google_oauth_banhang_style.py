# google_oauth_banhang_style.py
from __future__ import annotations

import os
import subprocess
import webbrowser
from pathlib import Path
from typing import List, Tuple, Optional

try:
    # Nếu BIENSO có Tkinter UI thì sẽ hiện popup
    from tkinter import messagebox
except Exception:
    messagebox = None

from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request


# ================== CẤU HÌNH GIỐNG BANHANG ==================

# 1) OAuth app credentials (NHÚNG TRỰC TIẾP)
CLIENT_ID = "604695997139-aac5bos2h35otasf2d15uf94g5r1juf5.apps.googleusercontent.com"
CLIENT_SECRET = "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"

# 2) Token lưu bền vững ở LOCALAPPDATA\Token\token.json (y như bạn gửi)
TOKEN_DIR = Path(os.environ.get("LOCALAPPDATA") or os.path.expandvars(r"%LOCALAPPDATA%")) / "Token"
TOKEN_DIR.mkdir(parents=True, exist_ok=True)
TOKEN_PATH = TOKEN_DIR / "token.json"

# 3) Scope Drive (đúng như đoạn code bạn gửi)
SCOPES: List[str] = ["https://www.googleapis.com/auth/drive"]

# 4) Buộc dùng đúng Chrome bạn chỉ định
EXACT_CHROME_EXE = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
os.environ["CHROME_EXE"] = EXACT_CHROME_EXE


# ================== TIỆN ÍCH HIỂN THỊ LỖI ==================

def _show_error(title: str, msg: str) -> None:
    if messagebox is not None:
        try:
            messagebox.showerror(title, msg)
            return
        except Exception:
            pass
    # fallback console
    print(f"[{title}] {msg}")


# ================== MỞ CHROME ẨN DANH / GUEST (PATCH WEBBROWSER) ==================

def _is_incognito_allowed() -> bool:
    """
    Kiểm tra policy IncognitoModeAvailability:
    - 0 hoặc không đặt: cho phép
    - 1: chặn incognito
    - 2: luôn ẩn danh
    Nếu không đọc được registry -> coi như cho phép (giống code bạn).
    """
    try:
        import winreg
        keys = [
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Policies\Google\Chrome"),
            (winreg.HKEY_CURRENT_USER,  r"SOFTWARE\Policies\Google\Chrome"),
        ]
        for root, sub in keys:
            try:
                with winreg.OpenKey(root, sub) as k:
                    val, _ = winreg.QueryValueEx(k, "IncognitoModeAvailability")
                    if val == 1:
                        return False
            except OSError:
                pass
    except Exception:
        pass
    return True


def _launch_chrome_clean(url: str) -> bool:
    chrome_exe = EXACT_CHROME_EXE

    if not os.path.exists(chrome_exe):
        _show_error(
            "Chrome không tìm thấy",
            f"Không tìm thấy Chrome tại:\n{chrome_exe}\n\nHãy kiểm tra lại EXACT_CHROME_EXE."
        )
        return False

    flags = ["--new-window"]
    flags.append("--incognito" if _is_incognito_allowed() else "--guest")

    try:
        subprocess.Popen(
            [chrome_exe, *flags, url],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception as e:
        _show_error("Lỗi mở Chrome", f"Không thể khởi chạy Chrome:\n{chrome_exe}\n\nChi tiết: {e}")
        return False


def _patch_webbrowser_to_incognito() -> Tuple:
    """
    Monkey-patch webbrowser để bất kỳ lệnh open nào cũng mở Chrome clean profile
    (incognito hoặc guest) -> giống y hệt BANHANG.
    """
    orig_open = webbrowser.open
    orig_open_new = webbrowser.open_new
    orig_open_new_tab = webbrowser.open_new_tab
    orig_get = webbrowser.get

    def _incog_open(url, new=1, autoraise=True):
        return _launch_chrome_clean(url)

    class _Controller:
        def open(self, url, new=1, autoraise=True): return _launch_chrome_clean(url)
        def open_new(self, url): return _launch_chrome_clean(url)
        def open_new_tab(self, url): return _launch_chrome_clean(url)

    webbrowser.open = _incog_open
    webbrowser.open_new = _incog_open
    webbrowser.open_new_tab = _incog_open
    webbrowser.get = (lambda name=None: _Controller())

    return orig_open, orig_open_new, orig_open_new_tab, orig_get


# ================== CORE: ENSURE CREDS (Y HỆT LOGIC) ==================

def ensure_creds(scopes: Optional[List[str]] = None) -> Credentials:
    """
    - Đọc token.json nếu có
    - Refresh nếu hết hạn và có refresh_token
    - Nếu không hợp lệ -> OAuth bằng flow.run_local_server(port=0)
      + patch webbrowser để mở Chrome incognito/guest
    - Lưu lại token.json
    """
    scopes = scopes or SCOPES

    creds: Optional[Credentials] = None

    if TOKEN_PATH.exists():
        try:
            creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), scopes)
        except Exception:
            creds = None

    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
        except Exception:
            creds = None

    if not creds or not creds.valid:
        client_config = {
            "installed": {
                "client_id": CLIENT_ID,
                "client_secret": CLIENT_SECRET,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": ["http://localhost"],
            }
        }
        flow = InstalledAppFlow.from_client_config(client_config, scopes)

        orig_open, orig_open_new, orig_open_new_tab, orig_get = _patch_webbrowser_to_incognito()
        try:
            creds = flow.run_local_server(
                port=0,                 # cổng động
                open_browser=True,
                access_type="offline",  # lấy refresh_token
                prompt="consent",       # ép consent để chắc chắn có refresh_token
                success_message="Đăng nhập thành công, bạn có thể đóng tab này.",
            )
        finally:
            # restore y như bạn gửi
            webbrowser.open = orig_open
            webbrowser.open_new = orig_open_new
            webbrowser.open_new_tab = orig_open_new_tab
            webbrowser.get = orig_get

        TOKEN_PATH.write_text(creds.to_json(), encoding="utf-8")

    return creds


# ================== HÀM TIỆN DỤNG: BUILD DRIVE SERVICE ==================

def get_drive_service() :
    creds = ensure_creds(SCOPES)
    return build("drive", "v3", credentials=creds)
