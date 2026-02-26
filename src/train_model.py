import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from data_preprocessing import load_data, preprocess

def train():
    df = load_data(r"C:\Users\MY\Desktop\customer-churn-analytics\data\churn_data.csv")
    df = preprocess(df)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:,1]

    print(classification_report(y_test, preds))
    print("ROC AUC:", roc_auc_score(y_test, proba))

    joblib.dump(model, "../models/churn_model.pkl")
    joblib.dump(X.columns.tolist(), "../models/features.pkl")

if __name__ == "__main__":
    train()