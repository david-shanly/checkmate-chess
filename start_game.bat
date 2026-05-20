@echo off
title Neural Chess Trainer Startup Script
echo ===================================================
echo 🧠 Starting Neural Chess Trainer ...
echo ===================================================

:: Ensure we are in the correct folder structure
cd /d "%~dp0"

:: Check if chess-frontend folder exists
if not exist "chess-frontend" (
    echo [ERROR] Could not find the 'chess-frontend' directory!
    echo Please make sure this .bat file is placed in the project root folder.
    pause
    exit /b
)

:: Navigate to frontend
cd chess-frontend

:: If node_modules is missing, run npm install
if not exist "node_modules" (
    echo [INFO] node_modules not found. Installing frontend dependencies...
    call npm install
)

:: Open Visual Studio Code in the parent project directory
echo [INFO] Opening Visual Studio Code...
start "" code ..

:: Start the React frontend server
echo [INFO] Starting the Development Server...
call npm start

pause
