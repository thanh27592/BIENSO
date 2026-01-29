@echo off
set APP_NAME=BIENSO_ANPR
set MAIN=main.py

REM Xoá output cũ (tuỳ chọn)
rmdir /s /q dist 2>nul

python -m nuitka %MAIN% ^
  --standalone ^
  --enable-plugin=pyside6 ^
  --windows-console-mode=disable ^
  --output-dir=dist ^
  --output-filename=%APP_NAME% ^
  --include-data-files=config.json=config.json ^
  --include-data-files=license_plate_detector.pt=license_plate_detector.pt ^
  --include-package-data=ultralytics ^
  --include-package-data=easyocr
  --icon=icon.ico