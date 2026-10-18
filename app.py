
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "food_demand_sample.csv"
MODEL_PATH = BASE_DIR / "model" / "food_demand_model.pkl"

st.set_page_config(
    page_title="Food Demand Predictor",
    page_icon="🍲",
    layout="wide"
)

st.title("🍲 AI-Based Food Demand Prediction")
st.write("Predict meal demand and help reduce potential food surplus.")
st.info("This prototype uses sample data for demonstration.")

# Load historical data
if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.stop()

df = pd.read_csv(DATA_PATH)

df["potential_surplus"] = (
    df["meals_prepared"] - df["meals_consumed"]
).clip(lower=0)

st.subheader("📊 Historical Food Data")

col1, col2 = st.columns(2)
col1.metric("Total Meals Prepared", f"{df['meals_prepared'].sum():,.0f}")
col2.metric("Total Meals Consumed", f"{df['meals_consumed'].sum():,.0f}")

chart = px.line(
    df,
    x="date",
    y=["meals_prepared", "meals_consumed"],
    labels={
        "value": "Number of Meals",
        "date": "Date",
        "variable": "Meal Type"
    },
    title="Daily Meal Trends"
)
st.plotly_chart(chart, use_container_width=True)

st.subheader("Potential Food Surplus")
st.metric(
    "Total Potential Surplus",
    f"{df['potential_surplus'].sum():,.0f} meals"
)
st.caption(
    "This is an estimate from sample data, not measured actual food waste."
)

st.subheader("🔎 Explore the Dataset")
st.dataframe(df.head(10), use_container_width=True)

# Prediction section
st.divider()
st.header("🤖 Predict Meal Demand")

if not MODEL_PATH.exists():
    st.error("Trained model not found. Run: python train_model.py")
    st.stop()

model = joblib.load(MODEL_PATH)

st.write("Enter the conditions for the day you want to estimate.")

with st.form("prediction_form"):
    day = st.selectbox(
        "Day of the week",
        ["Monday", "Tuesday", "Wednesday", "Thursday",
         "Friday", "Saturday", "Sunday"]
    )

    temperature = st.number_input(
        "Temperature (°C)",
        min_value=-10.0,
        max_value=55.0,
        value=30.0,
        step=1.0
    )

    rainfall = st.number_input(
        "Rainfall (mm)",
        min_value=0.0,
        max_value=500.0,
        value=0.0,
        step=1.0
    )

    event = st.selectbox(
        "Is there a special event?",
        ["No", "Yes"]
    )

    submitted = st.form_submit_button("Predict Meal Demand")

if submitted:
    input_data = pd.DataFrame([{
        "day_of_week": day,
        "temperature_c": temperature,
        "rainfall_mm": rainfall,
        "special_event": event
    }])

    prediction = max(0, round(float(model.predict(input_data)[0])))

    st.success(f"Estimated meals required: {prediction:,}")

    st.caption(
        "This is a model-based estimate, not a guaranteed demand figure. "
        "Results depend on the quality and representativeness of the sample data."
    )
