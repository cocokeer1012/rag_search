@echo off
chcp 65001 >nul
title 研报问答助手
cd /d %~dp0
echo 正在启动研报问答助手...
echo 启动后请用浏览器打开 http://127.0.0.1:8000
echo 停止服务：关闭本窗口或按 Ctrl+C
echo.
.venv\Scripts\python.exe -m uvicorn app:app --port 8000
pause
