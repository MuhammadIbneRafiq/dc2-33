#!/bin/bash

echo "===================================="
echo "London Crime Analysis Project Setup"
echo "===================================="
echo

echo "[1/4] Setting up Python backend..."
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
echo "Backend setup complete!"
echo

echo "[2/4] Setting up Node.js frontend..."
cd ../frontend
npm install
echo "Frontend setup complete!"
echo

echo "[3/4] Verifying data directories..."
cd ..
if [ ! -d "data" ]; then
    echo "ERROR: Data directory not found. Please ensure crime data is in the data/ folder."
    exit 1
fi
echo "Data directories verified!"
echo

echo "[4/4] Setup complete!"
echo
echo "To start the application:"
echo "1. Backend: cd backend && source venv/bin/activate && python app.py"
echo "2. Frontend: cd frontend && npm run dev"
echo
echo "Or run ./start.sh to start both servers automatically."
echo 