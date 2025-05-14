@echo off
echo Setting up the project...

echo Creating Python virtual environment...
cd backend
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate

echo Installing Python dependencies...
pip install -r requirements.txt

echo Installing Node.js dependencies...
cd ..\frontend
npm install

echo Starting the backend server...
start cmd /k "cd ..\backend && call venv\Scripts\activate && python app.py"

echo Waiting for backend to start...
timeout /t 5 /nobreak

echo Starting the frontend server...
start cmd /k "cd ..\frontend && npm run dev"

echo Setup complete! The application is now running.
echo Press Ctrl+C in each terminal window to stop the servers when finished. 