@echo off
chcp 65001 >nul
title 建立索引
cd /d %~dp0
echo 正在为 docs/ 下的语料建立索引（需要几分钟）...
echo.
.venv\Scripts\python.exe app.py build
echo.
pause
