@echo off
REM Quick setup script for Windows PowerShell

echo ================================================
echo AI Safety Models POC - Quick Setup
echo ================================================
echo.

echo [1/3] Checking Python installation...
python --version
if errorlevel 1 (
    echo ERROR: Python not found! Please install Python 3.8+
    pause
    exit /b 1
)
echo.

echo [2/3] Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo.

echo [3/3] Setup complete!
echo.
echo ================================================
echo Next Steps:
echo ================================================
echo.
echo Option 1: Run CLI Demo
echo   python cli_chat.py
echo.
echo Option 2: Run Evaluation
echo   python evaluate.py
echo.
echo Option 3: Test Individual Models
echo   python models/abuse_detection.py
echo   python models/crisis_detection.py
echo   python models/escalation_detection.py
echo   python models/content_filtering.py
echo.
echo ================================================
echo.
pause
