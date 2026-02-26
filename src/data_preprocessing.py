import pandas as pd

def load_data(path):
    return pd.read_csv(r"C:\Users\MY\Desktop\customer-churn-analytics\data\churn_data.csv")

def preprocess(df):
    df = df.copy()

    # Drop customer id if exists
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Convert TotalCharges to numeric
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Convert target to binary
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].map({"Yes":1, "No":0})

    # One-hot encoding
    df = pd.get_dummies(df, drop_first=True)

    return df