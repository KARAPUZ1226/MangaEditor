@echo off
chcp 65001 > NUL
title MangaEditor
echo ========================================
echo          Запуск MangaEditor
echo ========================================
cd /d "%~dp0"
.\venv\Scripts\python.exe -u -X utf8 main.py
if errorlevel 1 (
    echo.
    echo Произошла ошибка при работе приложения.
    pause
)
