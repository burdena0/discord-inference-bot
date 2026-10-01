@echo off
rem starts the bot in its own minimized window (with auto-restart).
cd /d "%~dp0"
start "discord-inference-bot" /min cmd /c run-bot.bat
