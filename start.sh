#!/bin/bash

echo "===================================="
echo "London Crime Analysis Project"
echo "Starting Backend and Frontend..."
echo "===================================="
echo

echo "Starting backend server..."
cd backend
source venv/bin/activate
python app.py &
BACKEND_PID=$!
cd ..

echo "Starting frontend server..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo
echo "Both servers are starting..."
echo "Backend will be available at: http://localhost:5000"
echo "Frontend will be available at: http://localhost:5173"
echo
echo "Press Ctrl+C to stop both servers..."

# Function to handle cleanup on script exit
cleanup() {
    echo
    echo "Stopping servers..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo "Servers stopped."
    exit 0
}

# Set up trap to call cleanup function on script exit
trap cleanup INT TERM

# Wait for processes
wait $BACKEND_PID $FRONTEND_PID 