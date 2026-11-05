@echo off
rem Runs the bot and restarts it 30s after any exit (crash, or the watchdog
rem exiting a hung bot). To stop: Ctrl+C, then answer Y, or close the window.
cd %~dp0nba-prop-bot
:loop
venv\Scripts\python.exe main.py full_run
echo Bot exited (code %errorlevel%) - restarting in 30s. Ctrl+C to stop.
rem ping instead of timeout: timeout fails when output is redirected (run_bot.ps1)
ping -n 31 127.0.0.1 >nul
goto loop
