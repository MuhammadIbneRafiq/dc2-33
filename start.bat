@echo off
echo ====================================
echo London Crime Analysis Project
echo Starting Backend and Frontend...
echo ====================================
echo.

echo Starting backend server...
start "Backend Server" cmd /k "cd backend && venv\Scripts\activate && python app.py"

echo Starting frontend server...
start "Frontend Server" cmd /k "cd frontend && npm run dev"

echo.
echo Both servers are starting...
echo Backend will be available at: http://localhost:5000
echo Frontend will be available at: http://localhost:5173
echo.
echo Press any key to close this window...
pause > nul 