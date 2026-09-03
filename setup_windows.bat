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

echo [1/3] Creating Python Virtual Environment (venv)...
if not exist venv (
    python -m venv venv
)
call venv\Scripts\activate.bat

echo [2/3] Installing Python Backend Dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo [3/3] Installing Frontend Dependencies...
cd frontend
call npm install
cd ..

echo.
echo ===================================================
echo   Setup Complete!
echo   You can now start the workbench by running:
echo     start_windows.bat
echo ===================================================
pause
