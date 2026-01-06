import streamlit as st
import pandas as pd
from prophet import Prophet

st.title("📈 Time Series Forecasting Dashboard")

# 1️⃣ CSV upload
file = st.file_uploader("Upload your CSV (must have Date + Value)", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.write("Raw Data", df.head())

    # Make sure CSV has at least two columns: Date + Value
    if len(df.columns) < 2:
        st.error("CSV must have at least 2 columns: Date and numeric value")
    else:
        # Rename columns for Prophet
        df = df.rename(columns={df.columns[0]: "ds", df.columns[1]: "y"})
        df["ds"] = pd.to_datetime(df["ds"])

        st.success("📊 Training AI model...")

        # 2️⃣ Train Prophet model
        model = Prophet()
        model.fit(df)

        # 3️⃣ Select how many days to forecast
        days = st.slider("Forecast days", 7, 60, 30)

        future = model.make_future_dataframe(periods=days)
        forecast = model.predict(future)

        # 4️⃣ Show forecast chart
        st.subheader("🔮 Forecast Chart")
        st.line_chart(forecast.set_index("ds")["yhat"])

        # Show forecasted data
        st.subheader("Forecast Data")
        st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail())
