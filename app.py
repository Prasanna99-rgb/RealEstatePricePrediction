import streamlit as st
import numpy as np
import pickle
import pandas as pd

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="California Housing Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    try:
        with open("XGBR.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

model = load_model()

# ---------------- HEADER ----------------
st.title("🏠 California Housing Price Predictor")
st.markdown("""
Predict **median house prices** using ML (XGBoost).

Adjust the features and click **Predict** to get results instantly.
""")

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("ℹ️ Model Info")
    st.markdown("""
    - Model: **XGBoost Regressor**
    - Dataset: California Housing
    - Features: 8
    - Accuracy: ~80% R²
    """)
    st.divider()
    st.info("Tip: Higher income & better location → higher prices 📈")

# ---------------- INPUT UI ----------------
col1, col2 = st.columns(2)

with col1:
    medinc = st.slider("💰 Median Income", 0.0, 15.0, 3.0, 0.1)
    houseage = st.slider("🏚️ House Age", 0.0, 100.0, 20.0, 1.0)
    averooms = st.slider("🛏️ Avg Rooms", 1.0, 20.0, 5.0, 0.1)
    avebedrms = st.slider("🛌 Avg Bedrooms", 0.5, 10.0, 2.0, 0.1)

with col2:
    population = st.number_input("👥 Population", 0, 50000, 1000, 100)
    aveoccup = st.slider("👪 Avg Occupancy", 1.0, 10.0, 3.0, 0.1)
    latitude = st.slider("🌍 Latitude", 32.5, 42.0, 34.0, 0.01)
    longitude = st.slider("🌎 Longitude", -124.5, -114.0, -118.0, 0.01)

# ---------------- PREDICTION ----------------
if st.button("🔮 Predict House Price", use_container_width=True):

    try:
        input_data = np.array([[medinc, houseage, averooms, avebedrms,
                                population, aveoccup, latitude, longitude]])

        prediction = model.predict(input_data)[0] * 100_000

        # -------- DISPLAY RESULT --------
        st.success("✅ Prediction Successful!")

        st.metric(
            label="💵 Estimated House Price",
            value=f"${prediction:,.2f}"
        )

        # -------- INTERPRETATION --------
        if prediction > 500000:
            st.info("📈 High-value area (likely urban / coastal)")
        elif prediction > 200000:
            st.info("🏡 متوسط pricing area")
        else:
            st.info("📉 Affordable housing region")

        # -------- INPUT SUMMARY --------
        with st.expander("📋 Input Summary"):
            df = pd.DataFrame({
                "Feature": ["MedInc", "HouseAge", "AveRooms", "AveBedrms",
                            "Population", "AveOccup", "Latitude", "Longitude"],
                "Value": [medinc, houseage, averooms, avebedrms,
                          population, aveoccup, latitude, longitude]
            })
            st.dataframe(df, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit | ML Model: XGBoost")
