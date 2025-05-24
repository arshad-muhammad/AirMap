import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import os
import ee
import geemap
import folium
from streamlit_folium import st_folium
from datetime import datetime, timedelta
import tempfile
import rasterio
from rasterio.transform import from_bounds

# Set page config
st.set_page_config(page_title="AirMap", layout="wide")

# Initialize GEE
GEE_PROJECT = 'feedscan'  # Replace with your project ID
SERVICE_ACCOUNT_KEY = 'service-account-key.json'  # Path to service account key (if used)

try:
    if os.path.exists(SERVICE_ACCOUNT_KEY):
        credentials = ee.ServiceAccountCredentials('gee-service-account@project.iam.gserviceaccount.com', SERVICE_ACCOUNT_KEY)
        ee.Initialize(credentials=credentials, project=GEE_PROJECT)
    else:
        ee.Initialize(project=GEE_PROJECT)
except Exception:
    pass  # Silently proceed to fallback

# Title
st.title("AirMap")

# Check if .pkl files exist
pkl_files = {
    'data': 'gee_data.pkl',
    'no2_model': 'model_no2.pkl',
    'so2_model': 'model_so2.pkl',
    'o3_model': 'model_o3.pkl',
    'co_model': 'model_co.pkl',
    'aod_model': 'model_aod.pkl',
    'no2_scaler': 'scaler_no2.pkl',
    'so2_scaler': 'scaler_so2.pkl',
    'o3_scaler': 'scaler_o3.pkl',
    'co_scaler': 'scaler_co.pkl',
    'aod_scaler': 'scaler_aod.pkl'
}

missing_files = [name for name, file in pkl_files.items() if not os.path.exists(file)]
if missing_files:
    st.warning(f"Missing .pkl files: {', '.join(missing_files)}. Using fallback scaler for missing scalers.")
    for missing in missing_files:
        if 'scaler' in missing:
            pkl_files[missing] = None

# Load data and models
@st.cache_data
def load_data_and_models():
    data = pd.read_pickle(pkl_files['data'])
    models = {
        'NO2': pickle.load(open(pkl_files['no2_model'], 'rb')),
        'SO2': pickle.load(open(pkl_files['so2_model'], 'rb')),
        'O3': pickle.load(open(pkl_files['o3_model'], 'rb')),
        'CO': pickle.load(open(pkl_files['co_model'], 'rb')),
        'AOD': pickle.load(open(pkl_files['aod_model'], 'rb'))
    }
    scalers = {}
    for pollutant in models.keys():
        scaler_file = pkl_files.get(f'{pollutant.lower()}_scaler')
        if scaler_file and os.path.exists(scaler_file):
            scalers[pollutant] = pickle.load(open(scaler_file, 'rb'))
        else:
            scalers[pollutant] = StandardScaler().fit(data[feature_sets[pollutant]])
    
    for pollutant, model in models.items():
        features = feature_sets[pollutant]
        if all(f in data.columns for f in features):
            input_data = data[features].copy()
            input_data.fillna(input_data.mean(), inplace=True)
            input_scaled = scalers[pollutant].transform(input_data)
            data[f'fine_{pollutant}'] = model.predict(input_scaled)
        else:
            st.warning(f"Features {features} not found for {pollutant}. Setting fine_{pollutant} to 0.")
            data[f'fine_{pollutant}'] = 0
    return data, models, scalers

# Define feature sets
feature_sets = {
    'NO2': ['NO2_column_number_density', 'Optical_Depth_047'],
    'SO2': ['SO2_column_number_density', 'Optical_Depth_047'],
    'O3': ['O3_column_number_density', 'Optical_Depth_047'],
    'CO': ['CO_column_number_density', 'Optical_Depth_047'],
    'AOD': ['Optical_Depth_047', 'NO2_column_number_density']
}

data, models, scalers = load_data_and_models()

# Sidebar for navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Select Page", ["Data Overview", "Model Predictions", "Air Quality Maps", "AQI Analysis", "Environmental Alerts", "3D Air Quality Map","Create API"], index=0)

# AQI Calculation Function
def calculate_aqi(row):
    try:
        molar_masses = {'NO2': 46, 'SO2': 64, 'O3': 48, 'CO': 28}
        mixing_height = 1000
        concentrations = {}
        for pollutant in ['NO2', 'SO2', 'O3', 'CO']:
            col = f'{pollutant}_column_number_density'
            if col not in row or pd.isna(row[col]):
                concentrations[pollutant] = 0
            else:
                concentrations[pollutant] = row[col] * molar_masses[pollutant] * 1e6 / mixing_height
        
        concentrations['NO2_ppb'] = concentrations['NO2'] * 24.45 / molar_masses['NO2']
        concentrations['SO2_ppb'] = concentrations['SO2'] * 24.45 / molar_masses['SO2']
        concentrations['O3_ppb'] = concentrations['O3'] * 24.45 / molar_masses['O3']
        concentrations['CO_ppm'] = concentrations['CO'] * 24.45 / molar_masses['CO'] / 1000
        concentrations['PM2.5'] = row['Optical_Depth_047'] * 0.5 if 'Optical_Depth_047' in row and not pd.isna(row['Optical_Depth_047']) else 0
        
        breakpoints = {
            'NO2_ppb': [(0, 53, 0, 50), (54, 100, 51, 100), (101, 360, 101, 150), 
                        (361, 649, 151, 200), (650, 1249, 201, 300), (1250, 2049, 301, 500)],
            'SO2_ppb': [(0, 35, 0, 50), (36, 75, 51, 100), (76, 185, 101, 150), 
                        (186, 304, 151, 200), (305, 604, 201, 300), (605, 1004, 301, 500)],
            'O3_ppb': [(0, 54, 0, 50), (55, 70, 51, 100), (71, 85, 101, 150), 
                       (86, 105, 151, 200), (106, 200, 201, 300), (201, 500, 301, 500)],
            'CO_ppm': [(0, 4.4, 0, 50), (4.5, 9.4, 51, 100), (9.5, 12.4, 101, 150), 
                       (12.5, 15.4, 151, 200), (15.5, 30.4, 201, 300), (30.5, 50.4, 301, 500)],
            'PM2.5': [(0, 12.0, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150), 
                      (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300), (250.5, 500.4, 301, 500)]
        }
        
        aqi_values = {}
        for pollutant, conc in [('NO2_ppb', concentrations['NO2_ppb']), ('SO2_ppb', concentrations['SO2_ppb']),
                               ('O3_ppb', concentrations['O3_ppb']), ('CO_ppm', concentrations['CO_ppm']),
                               ('PM2.5', concentrations['PM2.5'])]:
            aqi = 0
            for c_low, c_high, aqi_low, aqi_high in breakpoints[pollutant]:
                if c_low <= conc <= c_high:
                    aqi = ((aqi_high - aqi_low) / (c_high - c_low)) * (conc - c_low) + aqi_low
                    break
            aqi_values[pollutant.split('_')[0] if '_' in pollutant else pollutant] = aqi
        
        overall_aqi = max(aqi_values.values()) if aqi_values else 0
        return overall_aqi, aqi_values, concentrations
    except Exception as e:
        st.error(f"Error calculating AQI: {e}. Ensure input data is valid.")
        return 0, {'NO2': 0, 'SO2': 0, 'O3': 0, 'CO': 0, 'PM2.5': 0}, {'PM2.5': 0, 'CO_ppm': 0}

# Wildfire Risk Calculation Function
def calculate_wildfire_risk(row, aqi, concentrations):
    try:
        pm25 = concentrations.get('PM2.5', 0)
        co_ppm = concentrations.get('CO_ppm', 0)
        
        pm25_thresholds = {'low': 35.4, 'high': 150.4}
        co_thresholds = {'low': 4.4, 'high': 9.4}
        aqi_thresholds = {'low': 50, 'high': 100}
        
        pm25_score = 0 if pm25 < pm25_thresholds['low'] else 1 if pm25 < pm25_thresholds['high'] else 2
        co_score = 0 if co_ppm < co_thresholds['low'] else 1 if co_ppm < co_thresholds['high'] else 2
        aqi_score = 0 if aqi < aqi_thresholds['low'] else 1 if aqi < aqi_thresholds['high'] else 2
        
        total_score = pm25_score + co_score + aqi_score
        
        if total_score <= 1:
            risk_level = "Low"
            color = "green"
        elif total_score <= 3:
            risk_level = "Moderate"
            color = "yellow"
        else:
            risk_level = "High"
            color = "red"
        
        factors = []
        if pm25_score > 0:
            factors.append(f"PM2.5: {pm25:.1f} µg/m³")
        if co_score > 0:
            factors.append(f"CO: {co_ppm:.1f} ppm")
        if aqi_score > 0:
            factors.append(f"AQI: {aqi:.1f}")
        
        return risk_level, color, factors, total_score
    except Exception as e:
        st.error(f"Error calculating wildfire risk: {e}. Check input data.")
        return "Unknown", "gray", [], 0

# Landing Page
if page == "Create API":
    st.header("Welcome to the API Creation Page")
    st.markdown("""
    Here you can just click the button "Generate API key" and get API keys to use wherever you want.
    
    ### Documentation

### 🔥 **1. Global Problem, Urgent Demand**

* **Air pollution** kills over **7 million people per year** globally (WHO).
* Cities and governments **urgently need local, real-time air quality insights**, but most tools are too broad or outdated.
* Businesses, especially in **health tech**, **climate tech**, and **insurance**, need this data for smarter decision-making.

### 💸 **2. Multi-Billion Dollar Market**

* The **air quality monitoring market** is expected to reach **\$6–10 billion by 2030**.
* **API-as-a-service** models (like weather or geolocation APIs) have already proven to be billion-dollar segments (e.g., OpenWeather, Mapbox).
* Governments worldwide are investing billions in **smart cities, sustainability**, and **environmental compliance**—all potential clients.

### 🧠 **3. Unique Product Advantages**

* Uses **AI to enhance and fill gaps** in raw satellite data—most current tools don’t do this well.
* Offers **hyperlocal, interactive 3D air quality mapping**, a huge leap over static pollution charts.
* Bundles **meteorological context**, **health impact analysis**, and **APIs**—creating a complete data platform, not just a dashboard.

### 🚀 **4. Multiple Revenue Streams**

1. **Subscription plans** for individuals, researchers, and companies.
2. **API monetization** for SaaS, insurance, health tech, climate startups.
3. **B2G contracts** with city, state, and national governments.
4. **Consulting/Insights** for urban planning, transport, agriculture, etc.

### 🌍 **5. Expansion Potential**

* **Health apps** (integrate AQI + health alerts).
* **Climate risk modeling** for finance and real estate.
* **Agriculture**, logistics, and travel planning (weather, wind, and air quality in one view).
* Possible integration into **IoT devices**, like air purifiers or home assistants.

    ### Get Started
    Stay Tuned!
    """)


# Data Overview Page
elif page == "Data Overview":
    st.header("Historical Air Quality Data")
    st.write("Data from GEE (Jan 2019 - Sep 2020) for Turkey")
    
    st.subheader("Raw Data")
    st.dataframe(data)
    
    st.subheader("Summary Statistics")
    st.write(data.describe())
    
    st.subheader("Time Series of Pollutants")
    fig = px.line(data, x='month', y=['NO2_column_number_density', 'SO2_column_number_density', 
                                     'O3_column_number_density', 'CO_column_number_density', 
                                     'Optical_Depth_047'], 
                  title="Pollutant Concentrations Over Time")
    st.plotly_chart(fig)
    
    st.subheader("Correlation Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', center=0, ax=ax)
    st.pyplot(fig)

# Model Predictions Page
elif page == "Model Predictions":
    st.header("Predict Fine-Resolution Pollutant Levels")
    st.write("Enter feature values to predict fine-resolution pollutant concentrations.")
    
    if 'inputs' not in st.session_state:
        st.session_state.inputs = {}
    if 'pollutant' not in st.session_state:
        st.session_state.pollutant = 'NO2'
    
    pollutant = st.selectbox("Select Pollutant", list(models.keys()), 
                             index=list(models.keys()).index(st.session_state.pollutant),
                             key='pollutant_select')
    st.session_state.pollutant = pollutant
    
    st.subheader(f"Input Features for {pollutant}")
    inputs = {}
    for feature in feature_sets[pollutant]:
        min_val = float(data[feature].min())
        max_val = float(data[feature].max())
        default_val = float(data[feature].mean())
        
        if feature not in st.session_state.inputs:
            st.session_state.inputs[feature] = default_val
        
        inputs[feature] = st.number_input(
            f"{feature} (Range: {min_val:.2e} to {max_val:.2e}, mol/m²)",
            min_value=min_val,
            max_value=max_val,
            value=st.session_state.inputs[feature],
            step=(max_val - min_val) / 100,
            key=f"input_{feature}",
            on_change=lambda: st.session_state.inputs.update({feature: st.session_state[f"input_{feature}"]})
        )
    
    if st.button("Predict"):
        try:
            input_data = np.array([inputs[feature] for feature in feature_sets[pollutant]]).reshape(1, -1)
            input_scaled = scalers[pollutant].transform(input_data)
            prediction = models[pollutant].predict(input_scaled)[0]
            st.success(f"Predicted Fine {pollutant}: {prediction:.4f} µg/m³")
            
            st.subheader("Historical vs Predicted")
            if f'fine_{pollutant}' in data.columns and not data[f'fine_{pollutant}'].isna().all():
                fig = px.line(data, x='month', y=f'fine_{pollutant}', title=f"Fine {pollutant} Over Time (µg/m³)")
                fig.add_scatter(x=[data['month'].iloc[-1]], y=[prediction], mode='markers', 
                                name='Prediction', marker=dict(size=10, color='red'))
                st.plotly_chart(fig)
            else:
                st.warning(f"No valid historical fine {pollutant} data available for plotting. Check data or model.")
        except Exception as e:
            st.error(f"Prediction failed: {e}. Ensure inputs are valid and within range.")
    
    st.subheader("Model Reliability")
    st.write("Model performance metrics are not available. Assuming R² > 0.8 based on typical XGBoost performance. For precise metrics, check training logs or re-run AirQualityDownscaling_GEE.ipynb.")

# Air Quality Maps Page
elif page == "Air Quality Maps":
    st.header("Fine-Resolution Air Quality Maps")
    st.write("Generate fine-resolution air quality maps for a selected pollutant and month.")
    
    valid_aod_months = data[data['Optical_Depth_047'].notna()]['month'].dt.strftime('%Y-%m').unique()
    st.write(f"Months with valid Optical_Depth_047 data: {', '.join(valid_aod_months)}")
    st.warning("Due to persistent issues with MODIS data, Optical_Depth_047 will use the mean value from gee_data.pkl.")
    
    pollutant = st.selectbox("Select Pollutant", list(models.keys()))
    months = data['month'].dt.strftime('%Y-%m').unique()
    default_month = '2020-01' if '2020-01' in months else months[0]
    month = st.selectbox("Select Month", months, index=list(months).index(default_month) if default_month in months else 0)
    
    roi_small = ee.Geometry.Rectangle([28, 38, 30, 40])
    scale = 20000
    
    datasets = {
        'NO2': ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_NO2').select('NO2_column_number_density'),
        'SO2': ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_SO2').select('SO2_column_number_density'),
        'O3': ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_O3').select('O3_column_number_density'),
        'CO': ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_CO').select('CO_column_number_density'),
        'AOD': ee.ImageCollection('MODIS/006/MCD19A2_GRANULES').select('Optical_Depth_047')
    }
    
    if st.button("Generate Map"):
        with st.spinner("Generating map..."):
            try:
                start_date = f"{month}-01"
                end_date = (datetime.strptime(start_date, '%Y-%m-%d') + 
                           pd.offsets.MonthEnd(0)).strftime('%Y-%m-%d')
                feature_images = {}
                mean_aod = data['Optical_Depth_047'].mean() if not pd.isna(data['Optical_Depth_047'].mean()) else 0.1
                for feature in feature_sets[pollutant]:
                    if feature == 'Optical_Depth_047':
                        continue
                    collection = datasets[feature.split('_')[0]].filterDate(start_date, end_date).filterBounds(roi_small)
                    collection_size = collection.size().getInfo()
                    if collection_size == 0:
                        raise ValueError(f"No data available for {feature} in {month}.")
                    image = collection.mean().unmask(0)
                    bands = image.bandNames().getInfo()
                    if feature not in bands:
                        raise ValueError(f"Band {feature} not found in dataset. Available bands: {bands}")
                    feature_images[feature] = image
                
                try:
                    aod_collection = datasets['AOD'].filterDate(start_date, end_date).filterBounds(roi_small)
                    aod_size = aod_collection.size().getInfo()
                except:
                    pass
                
                feature_arrays = {}
                height, width = None, None
                for feature in feature_sets[pollutant]:
                    if feature == 'Optical_Depth_047':
                        if height is None or width is None:
                            other_feature = [f for f in feature_sets[pollutant] if f != feature][0]
                            other_array = geemap.ee_to_numpy(feature_images[other_feature], region=roi_small, scale=scale)
                            if other_array is None or other_array.size == 0:
                                raise ValueError(f"Cannot determine shape from {other_feature}.")
                            height, width = other_array.squeeze().shape
                        array = np.full((height, width), mean_aod)
                        feature_arrays[feature] = array
                        continue
                    try:
                        array = geemap.ee_to_numpy(feature_images[feature], region=roi_small, scale=scale)
                        if array is None or array.size == 0:
                            raise ValueError(f"Failed to retrieve data for {feature}.")
                        feature_arrays[feature] = array.squeeze()
                        if height is None or width is None:
                            height, width = feature_arrays[feature].shape
                    except ee.EEException:
                        raise  # Trigger fallback
                    except Exception as e:
                        raise ValueError(f"Unexpected error retrieving {feature}: {e}")
                
                shapes = [arr.shape for arr in feature_arrays.values()]
                if len(set(shapes)) > 1:
                    raise ValueError(f"Feature rasters have inconsistent shapes: {shapes}.")
                
                input_data = np.stack([feature_arrays[feature].flatten() for feature in feature_sets[pollutant]], axis=1)
                input_data[np.isnan(input_data)] = 0
                input_scaled = scalers[pollutant].transform(input_data)
                
                predictions = models[pollutant].predict(input_scaled).reshape(height, width)
                
                st.subheader(f"Fine {pollutant} Map for {month} (Static)")
                fig, ax = plt.subplots()
                im = ax.imshow(predictions, cmap='viridis', extent=[28, 30, 38, 40])
                plt.colorbar(im, ax=ax, label=f'Fine {pollutant} (µg/m³)')
                plt.title(f'Fine {pollutant} Map for {month}')
                plt.xlabel('Longitude')
                plt.ylabel('Latitude')
                st.pyplot(fig)
                
                try:
                    bounds = [28, 38, 30, 40]
                    transform = from_bounds(*bounds, width=width, height=height)
                    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp_file:
                        with rasterio.open(
                            tmp_file.name, 'w', driver='GTiff',
                            height=height, width=width, count=1,
                            dtype=predictions.dtype, crs='EPSG:4326',
                            transform=transform
                        ) as dst:
                            dst.write(predictions, 1)
                    
                    m = folium.Map(location=[39, 29], zoom_start=8)
                    folium.RasterImage(
                        tmp_file.name,
                        bounds=[[38, 28], [40, 30]],
                        colormap='viridis',
                        opacity=0.7
                    ).add_to(m)
                    st.subheader(f"Fine {pollutant} Map for {month} (Interactive)")
                    st_folium(m, width=700, height=500)
                    
                    os.remove(tmp_file.name)
                
                except Exception as e:
                    st.warning(f"Failed to create GeoTIFF or Folium map: {e}. Static map displayed above.")
                
            except Exception:
                try:
                    month_data = data[data['month'].dt.strftime('%Y-%m') == month]
                    if not month_data.empty:
                        mean_values = month_data[feature_sets[pollutant]].mean().to_numpy().reshape(1, -1)
                        mean_values[np.isnan(mean_values)] = 0
                        input_scaled = scalers[pollutant].transform(mean_values)
                        prediction = models[pollutant].predict(input_scaled)[0]
                        st.success(f"Predicted Fine {pollutant} (mean for {month}): {prediction:.4f} µg/m³")
                        height, width = 50, 50
                        uniform_map = np.full((height, width), prediction)
                        fig, ax = plt.subplots()
                        im = ax.imshow(uniform_map, cmap='viridis', extent=[28, 30, 38, 40])
                        plt.colorbar(im, ax=ax, label=f'Fine {pollutant} (µg/m³)')
                        plt.title(f"Fallback Fine {pollutant} Map for {month} (Uniform)")
                        plt.xlabel('Longitude')
                        plt.ylabel('Latitude')
                        st.pyplot(fig)
                    else:
                        st.error(f"No data for {month} in gee_data.pkl.")
                except Exception as e:
                    st.error(f"Fallback failed: {e}. Check gee_data.pkl or try another month.")

# AQI Analysis Page
elif page == "AQI Analysis":
    st.header("Air Quality Index (AQI) Analysis")
    st.write("Calculate and visualize AQI based on pollutant concentrations from gee_data.pkl.")
    
    aqi_data = data.copy()
    aqi_data[['AQI', 'AQI_breakdown', 'concentrations']] = aqi_data.apply(calculate_aqi, axis=1, result_type='expand')
    
    months = aqi_data['month'].dt.strftime('%Y-%m').unique()
    default_month = '2020-01' if '2020-01' in months else months[0]
    selected_month = st.selectbox("Select Month", months, index=list(months).index(default_month) if default_month in months else 0)
    
    month_data = aqi_data[aqi_data['month'].dt.strftime('%Y-%m') == selected_month].iloc[0]
    aqi = month_data['AQI']
    aqi_breakdown = month_data['AQI_breakdown']
    
    st.subheader(f"AQI for {selected_month}")
    aqi_category = ("Good" if aqi <= 50 else 
                    "Moderate" if aqi <= 100 else 
                    "Unhealthy for Sensitive Groups" if aqi <= 150 else 
                    "Unhealthy" if aqi <= 200 else 
                    "Very Unhealthy" if aqi <= 300 else "Hazardous")
    st.write(f"**AQI**: {aqi:.1f} ({aqi_category})")
    
    st.subheader("AQI Breakdown by Pollutant")
    breakdown_df = pd.DataFrame({
        'Pollutant': list(aqi_breakdown.keys()),
        'AQI': list(aqi_breakdown.values())
    })
    st.dataframe(breakdown_df)
    
    fig = px.bar(breakdown_df, x='Pollutant', y='AQI', title=f"AQI Breakdown for {selected_month}")
    st.plotly_chart(fig)
    
    st.subheader("AQI Over Time")
    fig = px.line(aqi_data, x='month', y='AQI', title="AQI Trend (Jan 2019 - Sep 2020)")
    st.plotly_chart(fig)

# Environmental Alerts Page
elif page == "Environmental Alerts":
    st.header("Environmental Alerts")
    st.write("Assess wildfire risk based on air quality data from gee_data.pkl.")
    
    aqi_data = data.copy()
    aqi_data[['AQI', 'AQI_breakdown', 'concentrations']] = aqi_data.apply(calculate_aqi, axis=1, result_type='expand')
    aqi_data[['risk_level', 'risk_color', 'risk_factors', 'risk_score']] = aqi_data.apply(
        lambda row: calculate_wildfire_risk(row, row['AQI'], row['concentrations']), axis=1, result_type='expand')
    
    months = aqi_data['month'].dt.strftime('%Y-%m').unique()
    default_month = '2020-01' if '2020-01' in months else months[0]
    selected_month = st.selectbox("Select Month", months, index=list(months).index(default_month) if default_month in months else 0)
    
    month_data = aqi_data[aqi_data['month'].dt.strftime('%Y-%m') == selected_month].iloc[0]
    risk_level = month_data['risk_level']
    risk_color = month_data['risk_color']
    risk_factors = month_data['risk_factors']
    
    st.subheader(f"Wildfire Risk for {selected_month}")
    st.markdown(f"<h3 style='color: {risk_color};'>Risk Level: {risk_level}</h3>", unsafe_allow_html=True)
    
    if risk_factors:
        st.write("**Contributing Factors**:")
        for factor in risk_factors:
            st.write(f"- {factor}")
    else:
        st.write("No significant contributing factors.")
    
    st.subheader("Wildfire Risk Over Time")
    fig = px.line(aqi_data, x='month', y='risk_score', title="Wildfire Risk Score Trend (Jan 2019 - Sep 2020)",
                  labels={'risk_score': 'Risk Score'})
    st.plotly_chart(fig)

# 3D Air Quality Map Page
else:
    st.header("3D Air Quality Map")
    st.write("Interactive 3D visualization of all pollutants, AQI, wildfire risk, temperature, and animated wind speed for Turkey.")
    
    months = data['month'].dt.strftime('%Y-%m').unique()
    default_month = '2020-01' if '2020-01' in months else months[0]
    selected_month = st.selectbox("Select Month", months, index=list(months).index(default_month) if default_month in months else 0)
    
    st.subheader("Select Layers to Display")
    show_aqi = st.checkbox("AQI (3D Surface)", value=True)
    show_risk = st.checkbox("Wildfire Risk", value=True)
    show_temp = st.checkbox("Temperature (°C)", value=True)
    show_wind = st.checkbox("Wind Speed Animation (m/s)", value=True)
    
    if st.button("Generate 3D Map"):
        with st.spinner("Generating 3D map..."):
            try:
                roi_small = ee.Geometry.Rectangle([28, 38, 30, 40])
                scale = 20000
                grid_size = 50
                lon = np.linspace(28, 30, grid_size)
                lat = np.linspace(38, 40, grid_size)
                lon_grid, lat_grid = np.meshgrid(lon, lat)
                
                fig = go.Figure()
                colorscales = {
                    'NO2': 'Plasma',
                    'SO2': 'Magma',
                    'O3': 'Viridis',
                    'CO': 'Inferno',
                    'AOD': 'Cividis',
                    'AQI': 'RdYlGn_r',
                    'Temp': 'Hot'
                }
                
                datasets = {
                    'NO2': ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_NO2').select('NO2_column_number_density'),
                    'SO2': ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_SO2').select('SO2_column_number_density'),
                    'O3': ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_O3').select('O3_column_number_density'),
                    'CO': ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_CO').select('CO_column_number_density'),
                    'AOD': ee.ImageCollection('MODIS/006/MCD19A2_GRANULES').select('Optical_Depth_047'),
                    'ERA5': ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY').select(['temperature_2m', 'u_component_of_wind_10m', 'v_component_of_wind_10m'])
                }
                
                start_date = f"{selected_month}-01"
                end_date = (datetime.strptime(start_date, '%Y-%m-%d') + 
                           pd.offsets.MonthEnd(0)).strftime('%Y-%m-%d')
                
                pollutant_arrays = {}
                height, width = None, None
                mean_aod = data['Optical_Depth_047'].mean() if not pd.isna(data['Optical_Depth_047'].mean()) else 0.1
                
                for pollutant in models.keys():
                    feature_images = {}
                    for feature in feature_sets[pollutant]:
                        if feature == 'Optical_Depth_047':
                            continue
                        collection = datasets[feature.split('_')[0]].filterDate(start_date, end_date).filterBounds(roi_small)
                        collection_size = collection.size().getInfo()
                        if collection_size == 0:
                            raise ValueError(f"No data for {feature} in {selected_month}.")
                        image = collection.mean().unmask(0)
                        feature_images[feature] = image
                    
                    feature_arrays = {}
                    for feature in feature_sets[pollutant]:
                        if feature == 'Optical_Depth_047':
                            if height is None or width is None:
                                other_feature = [f for f in feature_sets[pollutant] if f != feature][0]
                                other_array = geemap.ee_to_numpy(feature_images[other_feature], region=roi_small, scale=scale)
                                if other_array is None or other_array.size == 0:
                                    raise ValueError(f"Cannot retrieve {other_feature}.")
                                height, width = other_array.squeeze().shape
                            feature_arrays[feature] = np.full((height, width), mean_aod)
                        else:
                            array = geemap.ee_to_numpy(feature_images[feature], region=roi_small, scale=scale)
                            if array is None or array.size == 0:
                                raise ValueError(f"Failed to retrieve {feature}.")
                            feature_arrays[feature] = array.squeeze()
                            if height is None or width is None:
                                height, width = feature_arrays[feature].shape
                    
                    shapes = [arr.shape for arr in feature_arrays.values()]
                    if len(set(shapes)) > 1:
                        raise ValueError(f"Inconsistent shapes: {shapes}.")
                    
                    input_data = np.stack([feature_arrays[feature].flatten() for feature in feature_sets[pollutant]], axis=1)
                    input_data[np.isnan(input_data)] = 0
                    input_scaled = scalers[pollutant].transform(input_data)
                    predictions = models[pollutant].predict(input_scaled).reshape(height, width)
                    pollutant_arrays[pollutant] = predictions
                
                aqi_array = np.zeros((height, width))
                risk_array = np.zeros((height, width))
                risk_color_array = np.full((height, width), 'gray', dtype=object)
                
                for i in range(height):
                    for j in range(width):
                        row = {
                            'NO2_column_number_density': pollutant_arrays.get('NO2', np.zeros((height, width)))[i,j],
                            'SO2_column_number_density': pollutant_arrays.get('SO2', np.zeros((height, width)))[i,j],
                            'O3_column_number_density': pollutant_arrays.get('O3', np.zeros((height, width)))[i,j],
                            'CO_column_number_density': pollutant_arrays.get('CO', np.zeros((height, width)))[i,j],
                            'Optical_Depth_047': mean_aod
                        }
                        aqi, _, concentrations = calculate_aqi(pd.Series(row))
                        risk_level, risk_color, _, risk_score = calculate_wildfire_risk(row, aqi, concentrations)
                        aqi_array[i,j] = aqi
                        risk_array[i,j] = risk_score
                        risk_color_array[i,j] = risk_color
                
                temp_array = np.zeros((height, width))
                wind_frames = []
                if show_temp or show_wind:
                    era5_collection = datasets['ERA5'].filterDate(start_date, end_date).filterBounds(roi_small)
                    collection_size = era5_collection.size().getInfo()
                    if collection_size == 0:
                        raise ValueError(f"No ERA5-LAND data for {selected_month}.")
                    if show_temp:
                        temp_image = era5_collection.select('temperature_2m').mean()
                        temp_array = geemap.ee_to_numpy(temp_image, region=roi_small, scale=scale)
                        if temp_array is None or temp_array.size == 0:
                            raise ValueError("Failed to retrieve temperature data.")
                        temp_array = temp_array.squeeze() - 273.15  # Convert K to °C
                    if show_wind:
                        dates = [datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=i) for i in range(5)]
                        for date in dates:
                            date_str = date.strftime('%Y-%m-%d')
                            daily_collection = era5_collection.filterDate(date_str, (date + timedelta(days=1)).strftime('%Y-%m-%d'))
                            if daily_collection.size().getInfo() == 0:
                                continue
                            u_array = geemap.ee_to_numpy(daily_collection.select('u_component_of_wind_10m').mean(), region=roi_small, scale=scale)
                            v_array = geemap.ee_to_numpy(daily_collection.select('v_component_of_wind_10m').mean(), region=roi_small, scale=scale)
                            if u_array is None or v_array is None:
                                continue
                            wind_frames.append({
                                'u': u_array.squeeze(),
                                'v': v_array.squeeze(),
                                'date': date_str
                            })
                
                from scipy.interpolate import griddata
                points = np.array([(i, j) for i in range(height) for j in range(width)])
                lon_points = np.linspace(0, height-1, height).repeat(width).reshape(height, width)[points[:,0], points[:,1]]
                lat_points = np.linspace(0, width-1, width).repeat(height).reshape(width, height).T[points[:,0], points[:,1]]
                lon_points = lon_points * 2 / (height-1) + 28
                lat_points = lat_points * 2 / (width-1) + 38
                
                for pollutant in pollutant_arrays:
                    values = pollutant_arrays[pollutant].flatten()
                    grid = griddata((lon_points, lat_points), values, (lon_grid, lat_grid), method='linear')
                    unit = 'µg/m³' if pollutant != 'AOD' else 'unitless'
                    fig.add_trace(go.Scatter3d(
                        x=lon_grid.flatten(),
                        y=lat_grid.flatten(),
                        z=np.zeros_like(grid.flatten()),
                        mode='markers',
                        marker=dict(
                            size=4,
                            color=grid.flatten(),
                            colorscale=colorscales[pollutant],
                            opacity=0.6,
                            colorbar=dict(
                                title=f'{pollutant} ({unit})',
                                x=0.8 + 0.05 * list(models.keys()).index(pollutant)
                            ),
                            showscale=True
                        ),
                        name=f'Fine {pollutant}',
                        hovertemplate=f'{pollutant}: %{{marker.color:.2f}} {unit}<br>Lon: %{{x:.2f}}°E<br>Lat: %{{y:.2f}}°N'
                    ))
                
                if show_aqi:
                    aqi_values = aqi_array.flatten()
                    aqi_grid = griddata((lon_points, lat_points), aqi_values, (lon_grid, lat_grid), method='linear')
                    fig.add_trace(go.Surface(
                        x=lon, y=lat, z=aqi_grid * 0.5,
                        colorscale=colorscales['AQI'],
                        name='AQI',
                        showscale=True,
                        opacity=0.7,
                        colorbar=dict(title='AQI', x=0.75),
                        hovertemplate='AQI: %{z:.1f}<br>Lon: %{x:.2f}°E<br>Lat: %{y:.2f}°N'
                    ))
                
                if show_temp and temp_array.size > 0:
                    temp_values = temp_array.flatten()
                    temp_grid = griddata((lon_points, lat_points), temp_values, (lon_grid, lat_grid), method='linear')
                    fig.add_trace(go.Surface(
                        x=lon, y=lat, z=temp_grid * 0.5 + 100,
                        colorscale=colorscales['Temp'],
                        name='Temperature',
                        showscale=True,
                        opacity=0.5,
                        colorbar=dict(title='Temp (°C)', x=0.70),
                        hovertemplate='Temp: %{z:.1f}°C<br>Lon: %{x:.2f}°E<br>Lat: %{y:.2f}°N'
                    ))
                
                if show_risk:
                    risk_values = risk_array.flatten()
                    risk_colors = risk_color_array.flatten()
                    risk_grid = griddata((lon_points, lat_points), risk_values, (lon_grid, lat_grid), method='linear')
                    fig.add_trace(go.Scatter3d(
                        x=lon_grid.flatten(),
                        y=lat_grid.flatten(),
                        z=np.zeros_like(risk_grid.flatten()) + 50,
                        mode='markers',
                        marker=dict(
                            size=4,
                            color=risk_colors,
                            opacity=0.4
                        ),
                        name='Wildfire Risk',
                        hovertemplate='Risk Score: %{marker.color}<br>Lon: %{x:.2f}°E<br>Lat: %{y:.2f}°N'
                    ))
                
                if show_wind and wind_frames:
                    for i, frame in enumerate(wind_frames):
                        u_grid = griddata((lon_points, lat_points), frame['u'].flatten(), (lon_grid, lat_grid), method='linear')
                        v_grid = griddata((lon_points, lat_points), frame['v'].flatten(), (lon_grid, lat_grid), method='linear')
                        speed = np.sqrt(u_grid**2 + v_grid**2)
                        visible = (i == 0)
                        fig.add_trace(go.Cone(
                            x=lon_grid.flatten(),
                            y=lat_grid.flatten(),
                            z=np.full_like(lon_grid.flatten(), 150),
                            u=u_grid.flatten(),
                            v=v_grid.flatten(),
                            w=np.zeros_like(u_grid.flatten()),
                            colorscale='Blues',
                            sizemode='scaled',
                            sizeref=0.1,
                            showscale=(i == 0),
                            colorbar=dict(title='Wind Speed (m/s)', x=0.65),
                            name=f'Wind ({frame["date"]})',
                            visible=visible,
                            hovertemplate='Wind Speed: %{customdata:.2f} m/s<br>Lon: %{x:.2f}°E<br>Lat: %{y:.2f}°N',
                            customdata=speed.flatten()
                        ))
                
                frames = []
                if show_wind and wind_frames:
                    for i, frame in enumerate(wind_frames):
                        frames.append(go.Frame(
                            data=[go.Cone(
                                x=lon_grid.flatten(),
                                y=lat_grid.flatten(),
                                z=np.full_like(lon_grid.flatten(), 150),
                                u=griddata((lon_points, lat_points), frame['u'].flatten(), (lon_grid, lat_grid), method='linear').flatten(),
                                v=griddata((lon_points, lat_points), frame['v'].flatten(), (lon_grid, lat_grid), method='linear').flatten(),
                                w=np.zeros_like(lon_grid.flatten()),
                                colorscale='Blues',
                                sizemode='scaled',
                                sizeref=0.1,
                                showscale=True,
                                colorbar=dict(title='Wind Speed (m/s)', x=0.65),
                                customdata=np.sqrt(
                                    griddata((lon_points, lat_points), frame['u'].flatten(), (lon_grid, lat_grid), method='linear')**2 +
                                    griddata((lon_points, lat_points), frame['v'].flatten(), (lon_grid, lat_grid), method='linear')**2
                                ).flatten()
                            )],
                            name=frame['date']
                        ))
                
                fig.update_layout(
                    title=f"3D Air Quality Map for {selected_month}",
                    scene=dict(
                        xaxis_title="Longitude (°E)",
                        yaxis_title="Latitude (°N)",
                        zaxis_title="Value",
                        aspectratio=dict(x=1, y=1, z=0.5),
                        camera=dict(eye=dict(x=1.5, y=1.5, z=1))
                    ),
                    width=1000,
                    height=700,
                    showlegend=True,
                    updatemenus=[{
                        'buttons': [
                            {
                                'args': [None, {'frame': {'duration': 1000, 'redraw': True}, 'fromcurrent': True}],
                                'label': 'Play',
                                'method': 'animate'
                            },
                            {
                                'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate'}],
                                'label': 'Pause',
                                'method': 'animate'
                            }
                        ],
                        'direction': 'left',
                        'pad': {'r': 10, 't': 87},
                        'showactive': False,
                        'type': 'buttons',
                        'x': 0.1,
                        'xanchor': 'right',
                        'y': 0,
                        'yanchor': 'top'
                    }],
                    sliders=[{
                        'steps': [
                            {
                                'args': [[frame.name], {'frame': {'duration': 1000, 'redraw': True}, 'mode': 'immediate'}],
                                'label': frame.name,
                                'method': 'animate'
                            } for frame in frames
                        ],
                        'x': 0.1,
                        'len': 0.9,
                        'xanchor': 'left',
                        'y': 0,
                        'yanchor': 'top'
                    }]
                )
                fig.frames = frames
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception:
                try:
                    month_data = data[data['month'].dt.strftime('%Y-%m') == selected_month]
                    if not month_data.empty:
                        aqi_data = month_data.copy()
                        aqi_data[['AQI', 'AQI_breakdown', 'concentrations']] = aqi_data.apply(calculate_aqi, axis=1, result_type='expand')
                        row = aqi_data.iloc[0]
                        aqi = row['AQI']
                        concentrations = row['concentrations']
                        
                        fig = go.Figure()
                        colorscales = {
                            'NO2': 'Plasma',
                            'SO2': 'Magma',
                            'O3': 'Viridis',
                            'CO': 'Inferno',
                            'AOD': 'Cividis',
                            'AQI': 'RdYlGn_r',
                            'Temp': 'Hot'
                        }
                        for pollutant in models.keys():
                            mean_values = month_data[feature_sets[pollutant]].mean().to_numpy().reshape(1, -1)
                            mean_values[np.isnan(mean_values)] = 0
                            input_scaled = scalers[pollutant].transform(mean_values)
                            prediction = models[pollutant].predict(input_scaled)[0]
                            unit = 'µg/m³' if pollutant != 'AOD' else 'unitless'
                            uniform_map = np.full((grid_size, grid_size), prediction)
                            fig.add_trace(go.Scatter3d(
                                x=lon_grid.flatten(),
                                y=lat_grid.flatten(),
                                z=np.zeros_like(uniform_map.flatten()),
                                mode='markers',
                                marker=dict(
                                    size=4,
                                    color=uniform_map.flatten(),
                                    colorscale=colorscales[pollutant],
                                    opacity=0.6,
                                    colorbar=dict(
                                        title=f'{pollutant} ({unit})',
                                        x=0.8 + 0.05 * list(models.keys()).index(pollutant)
                                    ),
                                    showscale=True
                                ),
                                name=f'Fine {pollutant} (Fallback)',
                                hovertemplate=f'{pollutant}: %{{marker.color:.2f}} {unit}<br>Lon: %{{x:.2f}}°E<br>Lat: %{{y:.2f}}°N'
                            ))
                        if show_aqi:
                            aqi_grid = np.full((grid_size, grid_size), aqi)
                            fig.add_trace(go.Surface(
                                x=lon, y=lat, z=aqi_grid * 0.5,
                                colorscale=colorscales['AQI'],
                                name='AQI (Fallback)',
                                showscale=True,
                                opacity=0.7,
                                colorbar=dict(title='AQI', x=0.75),
                                hovertemplate='AQI: %{z:.1f}<br>Lon: %{x:.2f}°E<br>Lat: %{y:.2f}°N'
                            ))
                        if show_risk:
                            _, risk_color, _, risk_score = calculate_wildfire_risk(row, aqi, concentrations)
                            risk_grid = np.full((grid_size, grid_size), risk_score)
                            fig.add_trace(go.Scatter3d(
                                x=lon_grid.flatten(),
                                y=lat_grid.flatten(),
                                z=np.zeros_like(risk_grid.flatten()) + 50,
                                mode='markers',
                                marker=dict(
                                    size=4,
                                    color=risk_color,
                                    opacity=0.4
                                ),
                                name='Wildfire Risk (Fallback)',
                                hovertemplate='Risk Score: %{marker.color}<br>Lon: %{x:.2f}°E<br>Lat: %{{y:.2f}}°N'
                            ))
                        if show_temp:
                            temp_grid = np.full((grid_size, grid_size), 15.0)
                            fig.add_trace(go.Surface(
                                x=lon, y=lat, z=temp_grid * 0.5 + 100,
                                colorscale=colorscales['Temp'],
                                name='Temperature (Fallback)',
                                showscale=True,
                                opacity=0.5,
                                colorbar=dict(title='Temp (°C)', x=0.70),
                                hovertemplate='Temp: %{z:.1f}°C<br>Lon: %{x:.2f}°E<br>Lat: %{{y:.2f}}°N'
                            ))
                        if show_wind:
                            for i, date in enumerate([datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=i) for i in range(5)]):
                                date_str = date.strftime('%Y-%m-%d')
                                u_grid = np.full((grid_size, grid_size), 1.0)
                                v_grid = np.full((grid_size, grid_size), 1.0)
                                speed = np.sqrt(u_grid**2 + v_grid**2)
                                visible = (i == 0)
                                fig.add_trace(go.Cone(
                                    x=lon_grid.flatten(),
                                    y=lat_grid.flatten(),
                                    z=np.full_like(lon_grid.flatten(), 150),
                                    u=u_grid.flatten(),
                                    v=v_grid.flatten(),
                                    w=np.zeros_like(u_grid.flatten()),
                                    colorscale='Blues',
                                    sizemode='scaled',
                                    sizeref=0.1,
                                    showscale=(i == 0),
                                    colorbar=dict(title='Wind Speed (m/s)', x=0.65),
                                    name=f'Wind ({date_str}) (Fallback)',
                                    visible=visible,
                                    hovertemplate='Wind Speed: %{customdata:.2f} m/s<br>Lon: %{x:.2f}°E<br>Lat: %{y:.2f}°N',
                                    customdata=speed.flatten()
                                ))
                        
                        frames = []
                        if show_wind:
                            for i, date in enumerate([datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=i) for i in range(5)]):
                                date_str = date.strftime('%Y-%m-%d')
                                u_grid = np.full((grid_size, grid_size), 1.0)
                                v_grid = np.full((grid_size, grid_size), 1.0)
                                speed = np.sqrt(u_grid**2 + v_grid**2)
                                frames.append(go.Frame(
                                    data=[go.Cone(
                                        x=lon_grid.flatten(),
                                        y=lat_grid.flatten(),
                                        z=np.full_like(lon_grid.flatten(), 150),
                                        u=u_grid.flatten(),
                                        v=v_grid.flatten(),
                                        w=np.zeros_like(u_grid.flatten()),
                                        colorscale='Blues',
                                        sizemode='scaled',
                                        sizeref=0.1,
                                        showscale=True,
                                        colorbar=dict(title='Wind Speed (m/s)', x=0.65),
                                        customdata=speed.flatten()
                                    )],
                                    name=date_str
                                ))
                        
                        fig.update_layout(
                            title=f"Fallback 3D Air Quality Map for {selected_month}",
                            scene=dict(
                                xaxis_title="Longitude (°E)",
                                yaxis_title="Latitude (°N)",
                                zaxis_title="Value",
                                aspectratio=dict(x=1, y=1, z=0.5),
                                camera=dict(eye=dict(x=1.5, y=1.5, z=1))
                            ),
                            width=1000,
                            height=700,
                            showlegend=True,
                            updatemenus=[{
                                'buttons': [
                                    {
                                        'args': [None, {'frame': {'duration': 1000, 'redraw': True}, 'fromcurrent': True}],
                                        'label': 'Play',
                                        'method': 'animate'
                                    },
                                    {
                                        'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate'}],
                                        'label': 'Pause',
                                        'method': 'animate'
                                    }
                                ],
                                'direction': 'left',
                                'pad': {'r': 10, 't': 87},
                                'showactive': False,
                                'type': 'buttons',
                                'x': 0.1,
                                'xanchor': 'right',
                                'y': 0,
                                'yanchor': 'top'
                            }],
                            sliders=[{
                                'steps': [
                                    {
                                        'args': [[frame.name], {'frame': {'duration': 1000, 'redraw': True}, 'mode': 'immediate'}],
                                        'label': frame.name,
                                        'method': 'animate'
                                    } for frame in frames
                                ],
                                'x': 0.1,
                                'len': 0.9,
                                'xanchor': 'left',
                                'y': 0,
                                'yanchor': 'top'
                            }]
                        )
                        fig.frames = frames
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error(f"No data for {selected_month} in gee_data.pkl.")
                except Exception as e:
                    st.error(f"Fallback failed: {e}. Check gee_data.pkl or try another month.")

# Footer
st.sidebar.write("Imagined & Built by [Sphere Hive](https://spherehive.com)")
