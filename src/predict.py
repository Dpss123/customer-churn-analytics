import joblib
import pandas as pd
from data_preprocessing import preprocess

def predict(input_df):
    model = joblib.load(r"C:\Users\MY\Desktop\customer-churn-analytics\models\churn_model.pkl")
    features = joblib.load(r"C:\Users\MY\Desktop\customer-churn-analytics\models\features.pkl")

    df = preprocess(input_df)

    # Align columns
    for col in features:
        if col not in df:
            df[col] = 0

    df = df[features]

    prob = model.predict_proba(df)[:,1]
    pred = (prob > 0.5).astype(int)

    return pred, prob