import streamlit as st
import pandas as pd
import numpy as np
from google.cloud import aiplatform
import os

from environs import Env

env = Env()
env.read_env()

# Set page configuration
st.set_page_config(
    page_title="Saudi Used Cars Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# Initialize Google Cloud credentials and endpoint
PROJECT_ID = env('PROJECT_ID')
ENDPOINT_ID = env('ENDPOINT_ID')
REGION = env('REGION')

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "cloud/dev_trial.json"

# Define constants
MAKES = ['Toyota', 'Honda', 'Mazda', 'Nissan', 'Ford', 'Chevrolet', 'Hyundai', 'Kia', 
         'BMW', 'Mercedes-Benz', 'Audi', 'Lexus']
REGIONS = ['Riyadh', 'Jeddah', 'Makkah', 'Al-Medina', 'Eastern Province', 'Asir', 
          'Tabuk', 'Al-Qassim']
OPTIONS = ['Standard', 'Semi Full', 'Full']
GEAR_TYPES = ['Automatic', 'Manual']
ORIGINS = ['Saudi', 'Gulf', 'American', 'Asian', 'European']

def generate_preprocessor():
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, OrdinalEncoder

    numeric_features = ['Mileage', 'Engine_Size']
    ordinal_feature = ['Options']
    binary_features = ['Gear_Type']
    low_card_features = ['Origin']
    high_card_features = ['Make', 'Type', 'Region']

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    year_transformer = Pipeline(steps=[
        ('scaler', MinMaxScaler())
    ])

    ordinal_transformer = Pipeline(steps=[
        ('ordinal', OrdinalEncoder(categories=[OPTIONS]))
    ])

    binary_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first', sparse_output=False))
    ])

    low_card_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(sparse_output=False))
    ])

    high_card_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('year', year_transformer, ['Year']),
            ('num', numeric_transformer, numeric_features),
            ('ord', ordinal_transformer, ordinal_feature),
            ('binary', binary_transformer, binary_features),
            ('low_card', low_card_transformer, low_card_features),
            ('high_card', high_card_transformer, high_card_features)
        ],
        remainder='drop'
    )

    return preprocessor

def predict_price(input_data):
    aiplatform.init(project=PROJECT_ID, location=REGION)
    endpoint = aiplatform.Endpoint(ENDPOINT_ID)
    
    preprocessor = generate_preprocessor()
    X_processed = preprocessor.fit_transform(input_data)
    
    prediction = endpoint.predict(instances=X_processed.tolist())
    return prediction.predictions[0]

def main():
    st.title("🚗 Saudi Used Cars Price Predictor")
    st.write("Enter the details of the car to predict its price")

    # Create two columns
    col1, col2 = st.columns(2)

    with col1:
        # Basic car information
        make = st.selectbox("Make", options=MAKES)
        car_type = st.text_input("Car Type/Model (e.g., Land Cruiser, Camry)")
        year = st.number_input("Year", min_value=1990, max_value=2024, value=2020)
        
        # Technical specifications
        engine_size = st.number_input("Engine Size (in liters)", 
                                    min_value=1.0, 
                                    max_value=8.0, 
                                    value=2.0, 
                                    step=0.1)
        mileage = st.number_input("Mileage (in kilometers)", 
                                 min_value=0, 
                                 max_value=500000, 
                                 value=50000)

    with col2:
        gear_type = st.selectbox("Transmission", options=GEAR_TYPES)
        options = st.selectbox("Options", options=OPTIONS)
        origin = st.selectbox("Origin", options=ORIGINS)
        region = st.selectbox("Region", options=REGIONS)
        negotiable = st.checkbox("Negotiable")

    # Create a button to make prediction
    if st.button("Predict Price", type="primary"):
        input_data = pd.DataFrame({
            'Type': [car_type],
            'Region': [region],
            'Make': [make],
            'Gear_Type': [gear_type],
            'Origin': [origin],
            'Options': [options],
            'Year': [year],
            'Engine_Size': [engine_size],
            'Mileage': [mileage],
            'Negotiable': [negotiable]
        })

        try:
            predicted_price = predict_price(input_data)
            
            st.success("Prediction Complete!")
            st.metric(
                label="Predicted Price (SAR)", 
                value=f"{predicted_price:,.2f}"
            )
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")


    with st.expander("ℹ️ About this predictor"):
        st.write("""
        This predictor uses machine learning to estimate the price of used cars in Saudi Arabia.
        The model takes into account various factors including:
        - Make and model of the car
        - Year of manufacture
        - Mileage
        - Engine size
        - Transmission type
        - Options package
        - Region of sale
        - Origin of the car
        
        The predictions are based on historical data of used car sales in Saudi Arabia.
                 
        Please check the model and data used in this repo: https://github.com/andresuchdata/saudi-used-cars
        """)

if __name__ == "__main__":
    main()