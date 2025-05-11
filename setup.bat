@echo off
echo Setting up DC2 Crime Analysis Application...

:: Check if Python is installed
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Python is not installed. Please install Python 3.8 or higher.
    exit /b 1
)

:: Check if Node.js is installed
where node >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Node.js is not installed. Please install Node.js 14 or higher.
    exit /b 1
)

:: Backend setup
echo Setting up backend...
cd backend || exit /b 1

:: Create virtual environment if it doesn't exist
if not exist venv (
    echo Creating Python virtual environment...
    python -m venv venv
)

:: Activate virtual environment
call venv\Scripts\activate

:: Install Python dependencies
echo Installing Python dependencies...
pip install -r requirements.txt

:: Return to root directory
cd ..

:: Frontend setup
echo Setting up frontend...
cd frontend || exit /b 1

:: Install Node.js dependencies
echo Installing Node.js dependencies...
call npm install

:: Return to root directory
cd ..

echo Setup complete!
echo.
echo To start the backend:
echo   cd backend
echo   venv\Scripts\activate
echo   python app.py
echo.
echo To start the frontend:
echo   cd frontend
echo   npm run dev
echo.
echo Enjoy using the DC2 Crime Analysis Application!

pause 