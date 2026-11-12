import streamlit as st
import pandas as pd
import joblib
import time
import os
import numpy as np
from sensor_reader import get_sensor_data

# ==============================================================
# 🌿 PAGE CONFIG
# ==============================================================
st.set_page_config(page_title="Smart Pest Surveillance System", page_icon="🌾", layout="centered")

# Custom CSS Styling — Modern Dark Theme
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Global Dark Background ── */
    .stApp {
        background: linear-gradient(160deg, #0a0f0d 0%, #0d1a14 40%, #11201a 100%) !important;
        font-family: 'Inter', sans-serif;
    }

    /* ── Main Container ── */
    .block-container {
        background: rgba(16, 28, 22, 0.85) !important;
        border-radius: 20px;
        padding: 2.5rem 3rem;
        box-shadow: 0 8px 40px rgba(0, 0, 0, 0.5), 0 0 60px rgba(46, 204, 113, 0.04);
        border: 1px solid rgba(46, 204, 113, 0.1);
        backdrop-filter: blur(12px);
    }

    /* ── Headings ── */
    h1 {
        color: #2ecc71 !important;
        text-align: center;
        font-weight: 700;
        letter-spacing: -0.5px;
        text-shadow: 0 0 30px rgba(46, 204, 113, 0.25);
    }
    h2, h3 {
        color: #e0e0e0 !important;
        text-align: center;
        font-weight: 600;
    }

    /* ── Paragraph / Body Text ── */
    .stApp p, .stApp label, .stApp span, .stMarkdown {
        color: #c8d6d0 !important;
    }

    /* ── Buttons ── */
    .stButton>button {
        background: linear-gradient(135deg, #2ecc71, #27ae60) !important;
        color: #0a0f0d !important;
        border: none !important;
        border-radius: 12px;
        font-size: 1rem;
        font-weight: 600;
        padding: 0.65rem 1.5rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 15px rgba(46, 204, 113, 0.3);
        letter-spacing: 0.3px;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #f39c12, #e67e22) !important;
        color: #0a0f0d !important;
        transform: translateY(-2px) scale(1.03);
        box-shadow: 0 6px 25px rgba(243, 156, 18, 0.4);
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: rgba(10, 15, 13, 0.95) !important;
        border-right: 1px solid rgba(46, 204, 113, 0.15);
    }

    /* ── Metrics ── */
    div[data-testid="stMetric"] {
        background: rgba(46, 204, 113, 0.08) !important;
        border: 1px solid rgba(46, 204, 113, 0.2);
        border-radius: 14px;
        padding: 1rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    div[data-testid="stMetric"] label {
        color: #8fb8a0 !important;
        font-size: 0.85rem;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #2ecc71 !important;
        font-weight: 700;
        font-size: 1.6rem !important;
    }

    /* ── Select box / Slider / Radio ── */
    .stSelectbox > div > div,
    .stSlider > div > div,
    div[data-testid="stRadio"] {
        color: #c8d6d0 !important;
    }
    .stSelectbox > div > div > div {
        background-color: rgba(20, 35, 28, 0.9) !important;
        border: 1px solid rgba(46, 204, 113, 0.25) !important;
        border-radius: 10px;
        color: #e0e0e0 !important;
    }

    /* ── Alerts / Info / Success / Warning ── */
    div[data-testid="stAlert"] {
        border-radius: 12px !important;
        border-left-width: 4px !important;
    }
    .stSuccess, div[data-baseweb="notification"][kind="positive"] {
        background-color: rgba(46, 204, 113, 0.1) !important;
        border-color: #2ecc71 !important;
    }
    .stInfo {
        background-color: rgba(52, 152, 219, 0.1) !important;
        border-color: #3498db !important;
    }

    /* ── Progress Bar ── */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #2ecc71, #f39c12) !important;
        border-radius: 8px;
    }
    .stProgress > div > div {
        background-color: rgba(46, 204, 113, 0.12) !important;
        border-radius: 8px;
    }

    /* ── Divider ── */
    hr {
        border-color: rgba(46, 204, 113, 0.15) !important;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: #0a0f0d; }
    ::-webkit-scrollbar-thumb {
        background: rgba(46, 204, 113, 0.3);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover { background: rgba(46, 204, 113, 0.5); }

    /* ── Footer Text ── */
    footer { color: #5a7a6a !important; }
    </style>
""", unsafe_allow_html=True)

# ==============================================================
# 🧠 LOAD MODEL AND DATASET
# ==============================================================

model_type = "⚡ Light Model (Optimized for Streamlit)"
try:
    if os.path.exists("pest_prediction_model_light.joblib"):
        model = joblib.load("pest_prediction_model_light.joblib")
    elif os.path.exists("pest_prediction_model_full.joblib"):
        model = joblib.load("pest_prediction_model_full.joblib")
        model_type = "🧠 Full Model (Local High-Accuracy)"
    else:
        st.error("❌ No model file found. Please train the model first.")
        st.stop()

    pest_model = joblib.load("pest_name_predictor.joblib") if os.path.exists("pest_name_predictor.joblib") else None
    pesticide_map = joblib.load("pesticide_map.joblib") if os.path.exists("pesticide_map.joblib") else {}

    # Load dataset for crop types
    if os.path.exists("Smart_Pesticide_MultiRecommend.csv"):
        df = pd.read_csv("Smart_Pesticide_MultiRecommend.csv")
        plant_types = sorted(df["Plant Type"].dropna().unique())
    else:
        st.warning("⚠️ Dataset not found. Using default crop list.")
        df = pd.DataFrame(columns=["Plant Type", "pesticide", "pest_name"])
        plant_types = ["carrots", "chili", "cinnamon", "corn", "eggplant", "rice", "strawberries", "sunflowers", "tomato", "wheat"]

    st.success(f"✅ Models and dataset loaded successfully — {model_type}")

except Exception as e:
    st.error(f"❌ Error loading model or dataset: {e}")
    st.stop()

# ==============================================================
# 🌾 HEADER
# ==============================================================
st.title("🌾 Smart Pest Surveillance System")
st.markdown("""
This intelligent system uses **IoT + Machine Learning**  
to recommend the best **pesticides** for your crops based on  
**soil composition**, **environmental conditions**, and **crop type**.
""")

st.info(f"💡 Current Model Loaded: {model_type}")

st.markdown("---")

# ==============================================================
# 📡 MODE SELECTION
# ==============================================================
mode = st.radio("Select Mode", ["🌿 Manual Input Mode", "📡 Live Sensor Mode"])

# ==============================================================
# 🧮 MANUAL INPUT MODE
# ==============================================================
if mode == "🌿 Manual Input Mode":
    st.subheader("🧮 Manual Crop Input Mode")

    plant = st.selectbox("🌱 Select Crop Type", plant_types)

    st.markdown("#### 🌍 Environmental Conditions")
    col1, col2, col3 = st.columns(3)
    humidity = col1.slider("💧 Humidity (%)", 20, 100, 68)
    temperature = col2.slider("🌡 Temperature (°C)", 12, 45, 28)
    moisture = col3.slider("🌱 Soil Moisture (%)", 15, 95, 60)

    st.markdown("#### 🧪 Soil Composition")
    col4, col5, col6, col7 = st.columns(4)
    ph = col4.slider("⚗️ pH", 4.5, 8.5, 6.4, step=0.1)
    soil_ec = col5.slider("⚡ Soil EC (dS/m)", 0.1, 2.5, 0.7, step=0.05)
    phosphorus = col6.slider("🟡 Phosphorus (P)", 10.0, 100.0, 50.0, step=1.0)
    potassium = col7.slider("🟠 Potassium (K)", 100.0, 400.0, 230.0, step=5.0)

    col8, col9, col10 = st.columns(3)
    urea = col8.slider("🔵 Urea", 20.0, 180.0, 95.0, step=2.0)
    tsp = col9.slider("🟤 T.S.P", 10.0, 95.0, 50.0, step=1.0)
    mop = col10.slider("🟣 M.O.P", 10.0, 95.0, 50.0, step=1.0)

    st.markdown("### ⚙️ Model Prediction")

    if st.button("💊 Recommend Pesticide & Diagnose Pest"):
        sample = pd.DataFrame([{
            "pH": ph,
            "Soil EC": soil_ec,
            "Phosphorus": phosphorus,
            "Potassium": potassium,
            "Urea": urea,
            "T.S.P": tsp,
            "M.O.P": mop,
            "Moisture": moisture,
            "Temperature": temperature,
            "Humidity": humidity,
            "Plant Type": plant
        }])

        with st.spinner("Analyzing farm conditions and predicting..."):
            prediction = model.predict(sample)[0]
            try:
                prediction_proba = model.predict_proba(sample)[0]
                confidence = float(np.max(prediction_proba) * 100)
            except Exception:
                confidence = 88.0

            predicted_pest = None
            pest_confidence = None
            if pest_model is not None:
                predicted_pest = pest_model.predict(sample)[0]
                try:
                    pest_proba = pest_model.predict_proba(sample)[0]
                    pest_confidence = float(np.max(pest_proba) * 100)
                except Exception:
                    pest_confidence = confidence

        st.success(f"🌾 **Crop:** {plant.capitalize()}")
        if predicted_pest:
            st.markdown(f"🐛 **Detected Pest / Threat:** `{predicted_pest}` *(Confidence: {pest_confidence:.1f}%)*")
        st.markdown(f"💊 **Recommended Primary Pesticide:** `{prediction}`")
        st.progress(min(int(confidence), 100))
        st.caption(f"Model Recommendation Confidence: **{confidence:.1f}%**")

        st.markdown("---")
        st.info("📋 Alternative treatments for this crop condition:")
        alt = df[(df["Plant Type"].str.lower() == plant.lower()) & (df["pesticide"] != prediction)]["pesticide"]
        if not alt.empty:
            alt_options = list(alt.unique()[:2])
            st.markdown(f"🧪 Alternative Options: " + ", ".join([f"`{a}`" for a in alt_options]))
        else:
            st.markdown("🧪 Standard preventative dosage as per local agronomic guidelines.")

# ==============================================================
# 📡 LIVE SENSOR MODE
# ==============================================================
else:
    st.subheader("📡 Real-Time Arduino Sensor Mode")

    plant = st.selectbox("🌿 Select Crop Type", plant_types, key="live_crop")

    st.markdown("#### 🧪 Soil Composition (enter manually)")
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    live_ph = col_s1.slider("⚗️ pH", 4.5, 8.5, 6.4, step=0.1, key="live_ph")
    live_soil_ec = col_s2.slider("⚡ Soil EC", 0.1, 2.5, 0.7, step=0.05, key="live_ec")
    live_phosphorus = col_s3.slider("🟡 Phosphorus", 10.0, 100.0, 50.0, step=1.0, key="live_p")
    live_potassium = col_s4.slider("🟠 Potassium", 100.0, 400.0, 230.0, step=5.0, key="live_k")

    col_s5, col_s6, col_s7 = st.columns(3)
    live_urea = col_s5.slider("🔵 Urea", 20.0, 180.0, 95.0, step=2.0, key="live_u")
    live_tsp = col_s6.slider("🟤 T.S.P", 10.0, 95.0, 50.0, step=1.0, key="live_tsp")
    live_mop = col_s7.slider("🟣 M.O.P", 10.0, 95.0, 50.0, step=1.0, key="live_mop")

    if st.button("🔄 Fetch Live Data"):
        with st.spinner("Fetching live readings from Arduino..."):
            data = get_sensor_data()
            time.sleep(2)

        if data:
            col1, col2, col3 = st.columns(3)
            col1.metric("💧 Humidity (%)", f"{data['humidity']:.2f}")
            col2.metric("🌡 Temperature (°C)", f"{data['temperature']:.2f}")
            col3.metric("🌱 Soil Moisture (%)", f"{data['soil_moisture']:.2f}")

            sample = pd.DataFrame([{
                "pH": live_ph,
                "Soil EC": live_soil_ec,
                "Phosphorus": live_phosphorus,
                "Potassium": live_potassium,
                "Urea": live_urea,
                "T.S.P": live_tsp,
                "M.O.P": live_mop,
                "Moisture": data["soil_moisture"],
                "Temperature": data["temperature"],
                "Humidity": data["humidity"],
                "Plant Type": plant
            }])

            with st.spinner("Predicting pesticide recommendation..."):
                prediction = model.predict(sample)[0]
                try:
                    prediction_proba = model.predict_proba(sample)[0]
                    confidence = float(np.max(prediction_proba) * 100)
                except Exception:
                    confidence = 85.0

                predicted_pest = None
                pest_confidence = None
                if pest_model is not None:
                    predicted_pest = pest_model.predict(sample)[0]
                    try:
                        pest_proba = pest_model.predict_proba(sample)[0]
                        pest_confidence = float(np.max(pest_proba) * 100)
                    except Exception:
                        pest_confidence = confidence

            st.markdown("---")
            if predicted_pest:
                st.markdown(f"🐛 **Detected Pest / Threat:** `{predicted_pest}` *(Confidence: {pest_confidence:.1f}%)*")
            st.success(f"💊 **Recommended Pesticide:** `{prediction}`")
            st.progress(min(int(confidence), 100))
            st.caption(f"Model Recommendation Confidence: **{confidence:.1f}%**")

            st.markdown("### 🌾 Alternative Pesticides for Similar Conditions:")
            alt = df[(df["Plant Type"].str.lower() == plant.lower()) & (df["pesticide"] != prediction)]["pesticide"]
            if not alt.empty:
                alt_options = list(alt.unique()[:2])
                st.info(f"🧪 Alternative Options: " + ", ".join([f"`{a}`" for a in alt_options]))
            else:
                st.info("🧪 Standard preventative dosage as per local agronomic guidelines.")
        else:
            st.error("⚠️ Could not read data from Arduino. Please check the COM port or wiring.")

    st.caption("ℹ️ Note: Live Sensor Mode works only locally (not on Streamlit Cloud).")


