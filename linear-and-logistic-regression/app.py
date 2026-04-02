# app.py — Melbourne Housing Price Predictor
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
import os

# Set working directory to the file's location to ensure relative paths work
os.chdir(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = 'saved_models'

@st.cache_resource
def load_models():
    reg_model    = joblib.load(os.path.join(MODEL_DIR, 'regression_pipeline.pkl'))
    clf_model    = joblib.load(os.path.join(MODEL_DIR, 'classification_pipeline.pkl'))
    with open(os.path.join(MODEL_DIR, 'feature_cols.json')) as f:
        feature_cols = json.load(f)
    return reg_model, clf_model, feature_cols

# Error handling for missing artifacts
try:
    reg_model, clf_model, feature_cols = load_models()
except FileNotFoundError:
    st.error("Model artifacts not found. Please run the notebook cells in 'Part 11 — Save Models' first.")
    st.stop()

st.title('Melbourne Housing Price Predictor')
st.markdown('Built with scikit-learn + Streamlit')

col1, col2 = st.columns(2)
with col1:
    rooms      = st.slider('Rooms', 1, 10, 3)
    bedroom2   = st.slider('Bedrooms', 1, 10, 3)
    bathroom   = st.slider('Bathrooms', 1, 5, 1)
    distance   = st.slider('Distance from CBD (km)', 0.0, 50.0, 10.0)
with col2:
    car        = st.slider('Car spaces', 0, 5, 1)
    land_size  = st.number_input('Land size (m²)', 0, 10000, 500)
    bld_area   = st.number_input('Building area (m²)', 0, 1000, 120)
    year_built = st.slider('Year built', 1850, 2023, 1990)

prop_type  = st.selectbox('Property type', ['h', 'u', 't'])
method     = st.selectbox('Sale method', ['S', 'SP', 'PI', 'VB', 'SA'])
regionname = st.selectbox('Region', [
    'Southern Metropolitan', 'Northern Metropolitan',
    'Western Metropolitan', 'Eastern Metropolitan',
    'South-Eastern Metropolitan', 'Eastern Victoria',
    'Northern Victoria', 'Western Victoria',
])

if st.button('Predict Price'):
    # ── Replicate training feature engineering ────────────────────────
    raw = pd.DataFrame([{
        'Rooms': rooms, 'Distance': distance, 'Bedroom2': bedroom2,
        'Bathroom': bathroom, 'Car': car, 'Landsize': land_size,
        'BuildingArea': bld_area, 'YearBuilt': year_built,
        'Type': prop_type, 'Method': method, 'Regionname': regionname,
    }])

    raw['Rooms_per_Bath']   = raw['Rooms'] / (raw['Bathroom'] + 1)
    raw['Property_Age']     = 2023 - raw['YearBuilt']
    raw['Log_Landsize']     = np.log1p(raw['Landsize'])
    raw['Log_BuildingArea'] = np.log1p(raw['BuildingArea'])
    raw['Is_House']         = (raw['Type'] == 'h').astype(int)

    # One-hot encode exactly as during training
    raw = pd.get_dummies(raw, columns=['Type', 'Method', 'Regionname'])

    # Align to training schema: add missing dummy columns as 0, drop unseen ones
    for col in feature_cols:
        if col not in raw.columns:
            raw[col] = 0
    raw = raw[feature_cols]

    X_new = raw.values.astype(np.float32)

    # ── Predictions ───────────────────────────────────────────────────
    price_log  = reg_model.predict(X_new)[0]
    price_aud  = np.expm1(price_log)
    prob_above = clf_model.predict_proba(X_new)[0, 1]

    st.success(f'### Estimated Price: ${price_aud:,.0f} AUD')
    st.info(f'Probability of being above median price: {prob_above:.1%}')
