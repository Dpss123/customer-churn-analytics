import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

st.set_page_config(page_title="Customer Churn Intelligence Dashboard", layout="wide")

st.title("Customer Churn Intelligence & Retention Dashboard")
st.caption("AI-powered churn prediction with actionable business insights")

#  LOAD MODEL 
try:
    model = joblib.load(r"C:\Users\MY\Desktop\customer-churn-analytics\models\churn_model.pkl")
    features = joblib.load(r"C:\Users\MY\Desktop\customer-churn-analytics\models\features.pkl")
    st.success("Model loaded successfully")
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

#  FILE UPLOAD 
file = st.file_uploader("Upload Customer Dataset", type=["csv"])

if file:

    df = pd.read_csv(file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    #  PREPROCESS 
    df_proc = pd.get_dummies(df, drop_first=True)

    for col in features:
        if col not in df_proc:
            df_proc[col] = 0

    df_proc = df_proc[features]

    #  PREDICTIONS 
    probs = model.predict_proba(df_proc)[:, 1]
    df["Churn Probability"] = probs
    df["Risk Level"] = pd.cut(
        probs,
        bins=[0, 0.3, 0.6, 1],
        labels=["Low", "Medium", "High"]
    )

    #  KPI METRICS 
    col1, col2, col3 = st.columns(3)

    churn_rate = (df["Risk Level"] == "High").mean() * 100
    avg_prob = df["Churn Probability"].mean() * 100
    high_risk_count = (df["Risk Level"] == "High").sum()

    col1.metric("Predicted Churn Rate", f"{churn_rate:.2f}%")
    col2.metric("Average Risk Score", f"{avg_prob:.2f}%")
    col3.metric("High Risk Customers", high_risk_count)

    st.divider()

    #  VISUALS 
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Churn Probability Distribution")
        fig = px.histogram(df, x="Churn Probability", nbins=25, color="Risk Level")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Customer Risk Segments")
        fig2 = px.pie(df, names="Risk Level", hole=0.4)
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    #  FILTER 
    st.subheader("Filter Customers by Risk Level")

    risk_filter = st.multiselect(
        "Select Risk Level",
        options=["Low", "Medium", "High"],
        default=["High"]
    )

    filtered_df = df[df["Risk Level"].isin(risk_filter)]

    st.dataframe(filtered_df, use_container_width=True)

    #  DOWNLOAD 
    csv = filtered_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Filtered Results",
        csv,
        "churn_predictions.csv",
        "text/csv"
    )

    st.success("Prediction & analysis completed successfully")