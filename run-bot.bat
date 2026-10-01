@echo off
rem runs the bot and restarts it if it crashes. a clean exit (/shutdown fully_exit:true) ends the loop.
title discord-inference-bot
cd /d "%~dp0"
set "BOT_PYTHON=python"
if exist ".venv\Scripts\python.exe" set "BOT_PYTHON=%~dp0.venv\Scripts\python.exe"
:loop
"%BOT_PYTHON%" -X utf8 -u bot.py >> bot.log 2>&1
if %errorlevel%==0 goto :eof
echo %date% %time% bot stopped unexpectedly (exit code %errorlevel%), restarting in 5 seconds >> bot.log
timeout /t 5 /nobreak >nul
goto loop
