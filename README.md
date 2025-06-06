# London Burglary Risk Analysis

This project visualizes burglary risk across London at the LSOA (Lower Layer Super Output Area) level, using data from police reports and socioeconomic indicators.

## Features

- Interactive map showing London LSOAs colored by burglary risk level
- Data from real burglary incidents aggregated by LSOA
- Police allocation optimization for high-risk areas
- Time series analysis and forecasting of burglary trends
- Correlation analysis between socioeconomic factors and burglary rates

## Map View

The map displays London at the LSOA level with:
- Color-coded areas indicating burglary risk (Very Low to Very High)
- Clear LSOA boundaries without detailed street view
- Fast loading and responsive interface
- Ability to click on areas to see detailed information

## Setup Instructions

### Automatic Setup (Recommended)

#### Windows
1. Run `setup.bat` by double-clicking it or running it from the command prompt
2. The script will:
   - Create a Python virtual environment
   - Install backend dependencies
   - Install frontend dependencies
   - Start both the backend and frontend servers

#### Unix-based systems (macOS, Linux)
1. Make the setup script executable: `chmod +x setup.sh`
2. Run the script: `./setup.sh`
3. The script will set up and start both servers

### Manual Setup

#### Backend Setup
1. Navigate to the backend directory: `cd backend`
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Unix: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Start the server: `python app.py`

#### Frontend Setup
1. Navigate to the frontend directory: `cd frontend`
2. Install dependencies: `npm install`
3. Start the development server: `npm run dev`

## Accessing the Application

Once both servers are running:
1. Backend API will be available at: http://localhost:5000/api
2. Frontend will be available at: http://localhost:3000

## Data Sources

- Police recorded crime data for London boroughs
- London LSOA boundary shapefiles
- Indices of Multiple Deprivation (IMD) data
- Census data for socioeconomic indicators

## Technologies Used

- Backend: Flask (Python), pandas, scikit-learn, geopandas
- Frontend: React, TypeScript, Leaflet for mapping
- Data Processing: pandas, NumPy

## API Endpoints

The following API endpoints are available:

- `/api/lsoa/list` - Get a list of all LSOAs
- `/api/imd/lsoa/<lsoa_code>` - Get IMD (Index of Multiple Deprivation) data for a specific LSOA
- `/api/burglary/time-series` - Get historical burglary time series data
- `/api/burglary/forecast` - Get burglary forecasts
- `/api/police/optimize` - Get optimized police allocation
- `/api/emmie/scores` - Get EMMIE scores (Effect, Mechanism, Moderation, Implementation, Economic evaluation)
- `/api/burglary/correlation` - Get correlation analysis for burglary and socioeconomic factors

## License

This project is licensed under the MIT License.
