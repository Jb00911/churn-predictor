# 📉 Customer Churn Predictor

![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-006400?style=flat)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-F7931E?style=flat)
![Status](https://img.shields.io/badge/Status-Live-brightgreen?style=flat)

> **Live Demo →** [View on Streamlit Cloud](https://YOUR_APP_URL.streamlit.app)

An interactive ML web app that predicts customer churn probability. Upload your own customer CSV and get live predictions, risk levels, and business recommendations — powered by XGBoost trained on the IBM Telco dataset.

---

## ✨ Features

- **Model & EDA** — ROC curve, confusion matrix, churn distribution, tenure analysis
- **Live Predictor** — Upload any customer CSV → instant churn probabilities + risk badges
- **Download Results** — Export predictions as CSV with churn scores attached
- **Insights** — Top 20 churn drivers, risk segments by tenure & charges, business recommendations

---

## 🚀 Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/churn-predictor.git
cd churn-predictor
pip install -r requirements.txt

# Add dataset (download from Kaggle)
# https://www.kaggle.com/datasets/blastchar/telco-customer-churn
# Place WA_Fn-UseC_-Telco-Customer-Churn.csv in the project folder

streamlit run app.py
```

---

## 📁 Structure

```
churn-predictor/
├── app.py                  ← Entry point
├── requirements.txt
├── pages/
│   ├── eda.py              ← Model metrics + EDA charts
│   ├── predict.py          ← CSV upload + live predictions
│   └── insights.py         ← Feature importance + recommendations
└── utils/
    └── model.py            ← Training + caching
```

---

## 👤 Author

**Jibran Shahid** — MS Data Science · University of Central Punjab

[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-0A66C2?style=flat)](https://jb00911.github.io/jibranshahid.github.io)
[![Sales Dashboard](https://img.shields.io/badge/Sales_Dashboard-Live-FF4B4B?style=flat)](https://sales-dashboard-jb00911.streamlit.app/)
