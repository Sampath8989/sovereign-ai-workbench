@echo off
echo ===================================================
echo   Starting Sovereign AI Workbench
echo ===================================================
echo.

:: Start Backend in a separate window
echo [1/2] Launching Backend API on http://localhost:8000 ...
start "Sovereign AI Backend" cmd /k "call venv\Scripts\activate.bat && python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000"

:: Start Frontend in a separate window
echo [2/2] Launching Frontend UI on http://localhost:5173 ...
start "Sovereign AI Frontend" cmd /k "cd frontend && npm run dev -- --host 0.0.0.0"

echo.
echo ===================================================
echo   Both services are launching in separate windows!
echo   Open your browser at: http://localhost:5173
echo ===================================================
echo.
pause
