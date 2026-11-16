@echo off
chcp 65001 >nul
cd /d "%~dp0"
python -m lensplit
if errorlevel 1 pause
