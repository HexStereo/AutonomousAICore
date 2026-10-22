@echo off
title Launching Autonomous AI Core - Desktop Studio
cd /d "%~dp0"

set "APP_PATH=%~dp0desktop_app\index.html"
set "EDGE_EXE=C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

if exist "%EDGE_EXE%" (
    start "" "%EDGE_EXE%" --app="file:///%APP_PATH:\=/%" --window-size=1320,880
) else (
    start "" "%APP_PATH%"
)
exit /b 0
