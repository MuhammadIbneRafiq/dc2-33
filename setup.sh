#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}Setting up DC2 Crime Analysis Application...${NC}"

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo -e "${RED}Python is not installed. Please install Python 3.8 or higher.${NC}"
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo -e "${RED}Node.js is not installed. Please install Node.js 14 or higher.${NC}"
    exit 1
fi

# Backend setup
echo -e "${GREEN}Setting up backend...${NC}"
cd backend || exit 1

# Create and activate a virtual environment
echo "Creating virtual environment..."
python -m venv venv

# Activate the virtual environment (Unix-based systems)
source venv/bin/activate

# Install dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Return to root directory
cd ..

# Frontend setup
echo -e "${GREEN}Setting up frontend...${NC}"
cd frontend || exit 1

# Install node modules
echo "Installing Node.js dependencies..."
npm install

# Return to root directory
cd ..

# Start the servers in separate terminals
echo "Starting backend server..."
cd backend
python app.py &

# Wait for the backend to start
sleep 3

echo "Starting frontend server..."
cd frontend
npm run dev

# To stop both servers, press Ctrl+C

echo -e "${GREEN}Setup complete!${NC}"
echo -e "${BLUE}To start the backend:${NC}"
echo "  cd backend"
echo "  source venv/bin/activate (or venv/Scripts/activate on Windows)"
echo "  python app.py"
echo -e "${BLUE}To start the frontend:${NC}"
echo "  cd frontend"
echo "  npm run dev"

echo -e "${GREEN}Enjoy using the DC2 Crime Analysis Application!${NC}" 