# AirMap - AI Powered Air Quality Mapping

## Project Overview

AirMap is a comprehensive platform that provides hyperlocal, AI-enhanced air quality mapping and visualization. The platform combines satellite data, machine learning models, and interactive visualizations to deliver detailed insights into air quality, weather patterns, and environmental risks.

## Features

- **AI-Based Super Resolution**: Enhances low-quality satellite data into detailed air quality maps using deep learning models
- **Real-Time 3D Visualizations**: Explore air pollution, temperature, wind, and pressure in immersive 3D animated maps
- **Health & API Integration**: Understand causes and health impacts of pollutants and connect with API for data access
- **User Authentication**: Secure access to the platform using Firebase Authentication
- **Interactive Dashboard**: Streamlit-powered dashboard for data exploration and analysis

## Tech Stack

- **Frontend**: HTML, CSS (Tailwind CSS), JavaScript
- **Backend**: Python, Streamlit
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Visualization**: Plotly, Matplotlib, Seaborn, Folium
- **Earth Engine**: Google Earth Engine (GEE) via `earthengine-api` and `geemap`
- **Authentication**: Firebase Authentication
- **Geospatial Processing**: Rasterio

## Project Structure

```
airmap/
├── index.html              # Landing page with authentication
├── app.py                  # Main Streamlit application
├── gee_script.py           # Google Earth Engine initialization script
├── dataAnalysisTraining.ipynb  # Jupyter notebook for model training
├── service-account-key.json    # GEE service account key (not included in repo)
├── model_no2.pkl           # Trained model for NO2 prediction
├── model_so2.pkl           # Trained model for SO2 prediction
├── model_o3.pkl            # Trained model for O3 prediction
├── model_co.pkl            # Trained model for CO prediction
├── model_aod.pkl           # Trained model for AOD prediction
├── scaler_no2.pkl          # Scaler for NO2 features
├── scaler_so2.pkl          # Scaler for SO2 features
├── scaler_o3.pkl           # Scaler for O3 features
├── scaler_co.pkl           # Scaler for CO features
├── scaler_aod.pkl          # Scaler for AOD features
└── gee_data.pkl            # Preprocessed GEE data
```

## User Flow

### 1. Landing Page Experience
1. User visits the AirMap landing page
2. They explore the platform features, demo visualizations, and API information
3. When ready to access the full platform, they click "Get Started" or "Login"

### 2. Authentication Process
1. New users complete the signup form with email and password
2. Returning users login with their credentials
3. Firebase Authentication verifies user identity
4. Upon successful authentication, users are redirected to the Streamlit dashboard

### 3. Dashboard Navigation
1. Users land on the Data Overview page showing historical air quality data
2. Using the sidebar, they can navigate to different functional areas:
   - **Data Overview**: Explore historical data and statistics
   - **Model Predictions**: Input parameters to predict pollutant levels
   - **Air Quality Maps**: Generate detailed air quality visualizations
   - **AQI Analysis**: View Air Quality Index calculations and trends
   - **Environmental Alerts**: Check wildfire risk assessments
   - **3D Air Quality Map**: Interact with immersive 3D visualizations
   - **Create API**: Generate API keys for external applications

### 4. Data Exploration Workflow
1. Start with Data Overview to understand historical trends
2. Use Model Predictions to forecast specific pollutant levels
3. Generate Air Quality Maps to visualize spatial distribution
4. Check AQI Analysis to understand health implications
5. Review Environmental Alerts for risk assessment
6. Explore the 3D Air Quality Map for comprehensive visualization

### 5. API Integration
1. Navigate to the Create API section
2. Generate an API key
3. Use the provided documentation to integrate AirMap data into external applications
4. Access data programmatically using the endpoints described in the documentation

## Setup Instructions

### Prerequisites

1. Python 3.7+ installed
2. Google Earth Engine account with API access
3. Firebase account with a project set up
4. Required Python packages (see Installation section)

### Firebase Setup

1. Create a Firebase project at [Firebase Console](https://console.firebase.google.com/)
2. Enable Email/Password authentication:
   - Go to Authentication → Sign-in method
   - Enable Email/Password authentication
3. Get your Firebase configuration:
   - Go to Project Settings → General
   - Scroll down to "Your apps" section
   - Copy the Firebase configuration object

### Google Earth Engine Setup

1. Sign up for Google Earth Engine at [earthengine.google.com](https://earthengine.google.com/)
2. Create a service account and download the key file:
   - Go to Google Cloud Console → IAM & Admin → Service Accounts
   - Create a service account with Earth Engine permissions
   - Create and download a JSON key
   - Save as `service-account-key.json` in the project root

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/airmap.git
   cd airmap
   ```

2. Install required Python packages:
   ```bash
   pip install streamlit pandas numpy matplotlib seaborn plotly scikit-learn earthengine-api geemap folium streamlit-folium rasterio
   ```

3. Authenticate with Google Earth Engine (one-time setup):
   ```bash
   python gee_script.py
   ```

4. Update Firebase configuration in `index.html` with your own Firebase project details:
   ```javascript
   const firebaseConfig = {
     apiKey: "YOUR_API_KEY",
     authDomain: "YOUR_AUTH_DOMAIN",
     projectId: "YOUR_PROJECT_ID",
     storageBucket: "YOUR_STORAGE_BUCKET",
     messagingSenderId: "YOUR_MESSAGING_SENDER_ID",
     appId: "YOUR_APP_ID"
   };
   ```

### Running the Application

1. Start the landing page server:
   ```bash
   # Using Python's built-in HTTP server
   python -m http.server 8000
   ```

2. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

3. Open your browser and navigate to:
   - Landing page: http://localhost:8000
   - Streamlit app: http://localhost:8501

## Usage Guide

### Authentication Flow

1. Visit the landing page at http://localhost:8000
2. Click "Get Started"
3. If not logged in, you'll see the login/signup modal
4. Create an account or log in with existing credentials
5. After successful authentication, you'll be redirected to the Streamlit app

### Streamlit App Navigation

The Streamlit app has several pages accessible from the sidebar:

1. **Data Overview**: View historical air quality data and statistics
2. **Model Predictions**: Make predictions using trained models
3. **Air Quality Maps**: Generate fine-resolution air quality maps
4. **AQI Analysis**: Calculate and visualize Air Quality Index
5. **Environmental Alerts**: Assess wildfire risk based on air quality data
6. **3D Air Quality Map**: Interactive 3D visualization of pollutants
7. **Create API**: Generate API keys for data access

### Detailed Feature Usage

#### Data Overview
- View raw data table with pollutant measurements
- Examine summary statistics for all variables
- Explore time series charts of pollutant concentrations
- Analyze correlation matrices between different pollutants

#### Model Predictions
- Select a pollutant type (NO2, SO2, O3, CO, AOD)
- Input feature values within the specified ranges
- Generate predictions for fine-resolution pollutant levels
- View historical vs. predicted values on a time series chart

#### Air Quality Maps
- Select a month and pollutant type
- Generate static and interactive maps showing pollutant distribution
- Visualize spatial patterns of air quality across the selected regions

#### AQI Analysis
- View calculated Air Quality Index for selected time periods
- Examine AQI breakdown by contributing pollutants
- Track AQI trends over time with interactive charts
- Understand health implications based on AQI categories

#### Environmental Alerts
- Assess wildfire risk based on current air quality data
- View risk levels (Low, Moderate, High) with color coding
- Identify contributing factors to elevated risk
- Monitor risk score trends over time

#### 3D Air Quality Map
- Select time period and visualization layers
- Interact with 3D representation of multiple pollutants
- Toggle between different data layers (AQI, temperature, wind)
- Animate wind patterns to understand pollutant transport

#### Create API
- Generate personal API key for external applications
- View documentation for available endpoints
- Understand rate limits and usage guidelines
- Test API connectivity with sample requests

## Model Training

The models were trained using the `dataAnalysisTraining.ipynb` notebook:

1. Data was collected from Google Earth Engine
2. Features were preprocessed and normalized
3. XGBoost models were trained for each pollutant
4. Models and scalers were saved as pickle files

## API Documentation

The API provides access to hyperlocal environmental data:

### Data Types
- AQI & Pollutants
- Weather Layers
- Region Metadata

### Usage Options
- Rate-limited Free Tier
- Custom Plans for Enterprises
- OAuth2 Secured Access

## Troubleshooting

### Common Issues

1. **Firebase Authentication Issues**:
   - Check browser console for errors
   - Verify Firebase configuration
   - Ensure Email/Password authentication is enabled in Firebase Console

2. **Google Earth Engine Errors**:
   - Verify service account has proper permissions
   - Check if `service-account-key.json` is in the correct location
   - Ensure you've authenticated with GEE

3. **Missing .pkl Files**:
   - Run the training notebook to generate model files
   - Check file paths in `app.py`

4. **Streamlit App Not Loading**:
   - Check if all required packages are installed
   - Verify port 8501 is not in use by another application

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or collaborations, contact us at contact@airmap.ai

## Acknowledgments

- Sphere Hive for project support
- Google Earth Engine for satellite data access
- Streamlit for the interactive dashboard framework
