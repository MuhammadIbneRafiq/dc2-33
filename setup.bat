@echo off
echo ====================================
echo London Crime Analysis Project Setup
echo ====================================
echo.

echo [1/4] Setting up Python backend...
cd backend
python -m venv venv
call venv\Scripts\activate.bat
pip install -r requirements.txt
echo Backend setup complete!
echo.

echo [2/4] Setting up Node.js frontend...
cd ..\frontend
call npm install
echo Frontend setup complete!
echo.

echo [3/4] Verifying data directories...
cd ..
if not exist "data" (
    echo ERROR: Data directory not found. Please ensure crime data is in the data/ folder.
    pause
    exit /b 1
)
echo Data directories verified!
echo.

echo [4/4] Setup complete!
echo.
echo To start the application:
echo 1. Backend: cd backend && venv\Scripts\activate && python app.py
echo 2. Frontend: cd frontend && npm run dev
echo.
echo Or run start.bat to start both servers automatically.
echo.
pause 