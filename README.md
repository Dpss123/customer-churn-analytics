# Customer Churn Prediction & Retention Analytics Dashboard

An end-to-end Data Science project that predicts customer churn risk and provides actionable retention insights through an interactive analytics dashboard.

This project demonstrates the complete machine learning lifecycle including data preprocessing, model training, prediction, and deployment via an interactive dashboard built with Streamlit.



## Project Highlights

• Built a machine learning model to predict customer churn probability  
• Engineered preprocessing pipeline to handle categorical features and missing values  
• Designed a business-focused dashboard showing churn risk distribution and KPIs  
• Enabled filtering of high-risk customers for targeted retention strategies  
• Created a deployable and reproducible ML workflow  



## Tech Stack

Python  
Pandas, NumPy, Scikit-learn  
Plotly for visualization  
Streamlit for dashboard deployment  
Joblib for model persistence  



## Dashboard Features

• Customer churn probability scoring  
• Risk segmentation (Low, Medium, High)  
• KPI cards showing churn metrics  
• Interactive charts and filtering  
• Downloadable prediction results  


## Project Structure

customer-churn-analytics/
│
├── dashboard/              # Streamlit UI files
│   └── app.py
│
├── models/                 # Saved ML models & scalers
│   ├── model.pkl
│   └── features.pkl
│
├── src/                    # Training & preprocessing scripts
│
├── data/                   # Dataset files
│
├── requirements.txt
└── README.md



## How to Run

1. Clone the repository 

2. Install dependencies  
   pip install -r requirements.txt

3. Run the dashboard  
   streamlit run dashboard/app.py

   


## Business Impact

This solution helps organizations:

• Identify customers at risk of leaving  
• Prioritize retention campaigns  
• Reduce churn and improve customer lifetime value  
• Support data-driven decision making  


## Author

Dheerendra Pratap Singh  
Aspiring Data Scientist | AI Engineer | Analytics Professional