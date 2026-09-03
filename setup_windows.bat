@echo off
echo ===================================================
echo   Sovereign AI Workbench - Windows Setup Script
echo ===================================================
echo.

:: Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in your PATH.
    echo Please install Python 3.10, 3.11, or 3.12 from python.org and check 'Add Python to PATH'.
    pause
    exit /b 1
)

:: Check Node
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Node.js is not installed or not in your PATH.
    echo Please install Node.js LTS from nodejs.org.
    pause
    exit /b 1
)

echo [1/5] Creating Python Virtual Environment (venv)...
if not exist venv (
    python -m venv venv
)
call venv\Scripts\activate.bat

echo [2/5] Installing Python Backend Dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt huggingface_hub

echo [3/5] Installing Frontend Dependencies...
cd frontend
call npm install
cd ..

echo [4/5] Setting up Environment (.env)...
if not exist .env (
    if exist .env.example (
        copy .env.example .env >nul
        echo [OK] Created .env from .env.example
    )
)

echo [5/5] Downloading Local Models for Offline Inference...
python scripts\download_models.py --model recommended

echo.
echo ===================================================
echo   Setup Complete!
echo   You can now start the workbench by running:
echo     start_windows.bat
echo ===================================================
pause
