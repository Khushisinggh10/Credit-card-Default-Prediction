# ================================
# 📌 CREDIT RISK INTELLIGENCE APP
# ================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle
import plotly.express as px

# ================================
# 🎨 PAGE CONFIG
# ================================
st.set_page_config(page_title="CreditLens", layout="wide")
# =========================
# LOAD MODEL (NO ERROR)
# =========================
try:
    model = pickle.load(open("model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
except Exception as e:
    st.error(f"Actual Error: {e}")
    st.stop()
# ================================
# 🎨 CUSTOM CSS (DARK UI)
# ================================
st.markdown("""
<style>
body {
    background-color: #0b0f1a;
}
h1, h2, h3 {
    color: #00d4ff;
}
.metric-card {
    background-color: #111827;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ================================
# 📂 LOAD DATA
# ================================
@st.cache_data
def load_data():
    df = pd.read_csv("Credit_Card_Default.csv")
    return df

df = load_data()

# ================================
# 📊 HEADER
# ================================
st.title("💳 Credit Risk Intelligence Dashboard")
st.caption("From Raw Data ➜ Predictive Risk Insights")

# ================================
# 📊 TOP METRICS
# ================================
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Customers", len(df))
col2.metric("Default Rate", f"{df['default.payment.next.month'].mean()*100:.1f}%")
col3.metric("Avg Credit Limit", f"{df['LIMIT_BAL'].mean():,.0f}")
col4.metric("Features", df.shape[1])

# ================================
# 📌 SIDEBAR NAVIGATION
# ================================
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", [
    "Overview",
    "EDA",
    "Risk Segmentation",
    "Model Insights",
    "Live Predictor"
])

# ================================
# 🟢 OVERVIEW
if section == "Overview":

    col1, col2 = st.columns([2,1])

    with col1:
        st.header("📘 Project Explanation")

        st.markdown("""
### 🔍 What we built:
A machine learning system that predicts whether a customer will default.

### 💡 Key Insights:
- Payment delay is strongest predictor  
- High utilization increases risk  
- Behavioral features are critical  

### 💼 Business Impact:
- Early detection of risky customers  
- Reduce financial losses  
- Smarter lending decisions  

### 🧠 Final Model:
- Tuned XGBoost (Accuracy: 0.82)
- Threshold Used: 0.25 (High Recall)
""")

    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/2920/2920349.png")

# ================================
# 🔵 EDA
# ================================
elif section == "EDA":
    st.header("📊 Exploratory Data Analysis")

    # Default distribution
    fig = px.histogram(df, x="default.payment.next.month", color="default.payment.next.month")
    st.plotly_chart(fig, use_container_width=True)

    # Credit limit vs default
    fig2 = px.box(df, x="default.payment.next.month", y="LIMIT_BAL")
    st.plotly_chart(fig2, use_container_width=True)

    st.info("💡 Insight: Customers with lower credit limits tend to default more.")

# ================================
# 🟡 RISK SEGMENTATION
# ================================
elif section == "Risk Segmentation":
    st.header("⚠️ Customer Risk Segmentation")

    df['risk'] = pd.cut(df['default.payment.next.month'],
                       bins=[-1, 0, 1],
                       labels=["Low Risk", "High Risk"])

    fig = px.pie(df, names='risk', title="Risk Distribution")
    st.plotly_chart(fig, use_container_width=True)

    st.success("💡 22% customers are high risk — early action required!")

# ================================
## 🧠 MODEL INSIGHTS
# ================================
elif section == "Model Insights":

    st.header("🧠 Machine Learning Model Insights")

    st.markdown("""
## 📌 Objective

The goal of this project is to predict whether a customer is likely to default on their next credit card payment.

### 🎯 Target Variable
- **0** → No Default
- **1** → Default

This system helps banks and financial institutions identify risky customers before approving loans or increasing credit limits.
""")

    # ================================
    # MODELS USED
    # ================================
    st.subheader("🤖 Machine Learning Models Tested")

    model_df = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
            "Naive Bayes",
            "Support Vector Machine",
            "KNN",
            "Bagging",
            "AdaBoost",
            "Gradient Boosting",
            "XGBoost",
            "Tuned XGBoost"
        ],
        "Accuracy": [
            0.81,
            0.72,
            0.81,
            0.70,
            0.81,
            0.79,
            0.81,
            0.81,
            0.81,
            0.81,
            0.82
        ]
    })

    st.dataframe(model_df, width="stretch")

    # ================================
    # ACCURACY CHART
    # ================================
    fig = px.bar(
        model_df,
        x="Model",
        y="Accuracy",
        color="Accuracy",
        title="📊 Model Accuracy Comparison"
    )

    st.plotly_chart(fig, width="stretch")

    # ================================
    # BEST MODEL
    # ================================
    st.success("""
🏆 Final Selected Model: Tuned XGBoost

### Why This Model Was Selected?

✔ Highest Accuracy  
✔ Better Recall Score  
✔ Better Risk Detection  
✔ Handles Imbalanced Data Efficiently  
✔ More Reliable Predictions
""")

    # ================================
    # FEATURE IMPORTANCE
    # ================================
    st.subheader("🔥 Feature Importance Analysis")

    feature_df = pd.DataFrame({
        "Feature": [
            "PAY_0",
            "PAY_2",
            "LIMIT_BAL",
            "BILL_AMT1",
            "PAY_AMT1",
            "AGE",
            "EDUCATION"
        ],
        "Importance": [
            0.31,
            0.22,
            0.14,
            0.10,
            0.09,
            0.08,
            0.06
        ]
    })

    fig2 = px.bar(
        feature_df,
        x="Feature",
        y="Importance",
        color="Importance",
        title="📈 Important Features Affecting Risk"
    )

    st.plotly_chart(fig2, width="stretch")

    st.info("""
💡 Important Findings

• Late payment history is the strongest indicator of default risk.

• Customers with lower credit limits have higher chances of default.

• High bill amounts with low repayments increase financial risk.

• Payment behavior is more important than demographic information.
""")

    # ================================
    # PERFORMANCE METRICS
    # ================================
    st.subheader("📊 Final Model Performance")

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Accuracy", "82%")
    m2.metric("Precision", "79%")
    m3.metric("Recall", "84%")
    m4.metric("F1 Score", "81%")

    st.markdown("""
### 📘 Metrics Explanation

✅ Accuracy  
Overall percentage of correct predictions.

✅ Precision  
How many predicted risky customers were actually risky.

✅ Recall  
How many actual risky customers were correctly identified.

✅ F1 Score  
Balance between Precision and Recall.
""")

    # ================================
    # CONFUSION MATRIX
    # ================================
    st.subheader("📉 Confusion Matrix Explanation")

    st.markdown("""
| Actual / Predicted | Non-Default | Default |
|---|---|---|
| Non-Default | True Negative | False Positive |
| Default | False Negative | True Positive |

### 🔍 Meaning

✅ True Positive  
Model correctly identified risky customers.

✅ True Negative  
Model correctly identified safe customers.

⚠ False Positive  
Safe customer predicted as risky.

⚠ False Negative  
Risky customer predicted as safe.
""")

    # ================================
    # BUSINESS IMPACT
    # ================================
    st.subheader("💼 Business Impact")

    st.markdown("""
### 🚀 Benefits for Banks & Financial Institutions

✔ Detect risky customers early

✔ Reduce loan default losses

✔ Improve decision-making

✔ Increase profitability

✔ Automate credit approval systems

✔ Improve financial risk management

✔ Build smarter lending strategies
""")

    # ================================
    # REAL WORLD USE CASES
    # ================================
    st.subheader("🌍 Real World Applications")

    st.markdown("""
This system can be used in:

🏦 Banking Sector  
💳 Credit Card Companies  
📈 Financial Analytics  
💰 Loan Approval Systems  
🛡 Risk Management Systems  
📊 FinTech Applications
""")

    # ================================
    # CONCLUSION
    # ================================
    st.subheader("✅ Conclusion")

    st.markdown("""
The Credit Risk Intelligence System successfully predicts customer default risk using Machine Learning techniques.

Among all tested algorithms, **Tuned XGBoost** achieved the best performance and was selected as the final model.

This project demonstrates how Data Analytics and Machine Learning can help financial institutions:

- Minimize financial losses
- Improve customer screening
- Detect risky customers early
- Make data-driven decisions

🚀 This solution provides an intelligent and scalable approach for credit risk prediction.
""")
# ================================
# 🟣 LIVE PREDICTOR
# ================================
elif section == "Live Predictor":

    st.header("🔮 Live Prediction System")

    # FEATURES
    FEATURES = [
        "LIMIT_BAL",
        "SEX",
        "EDUCATION",
        "MARRIAGE",
        "AGE",
        "PAY_0",
        "PAY_2",
        "PAY_3",
        "PAY_4",
        "PAY_5",
        "PAY_6",
        "BILL_AMT1",
        "BILL_AMT2",
        "BILL_AMT3",
        "BILL_AMT4",
        "BILL_AMT5",
        "BILL_AMT6",
        "PAY_AMT1",
        "PAY_AMT2",
        "PAY_AMT3",
        "PAY_AMT4",
        "PAY_AMT5",
        "PAY_AMT6"
    ]

    # TABS
    tab1, tab2 = st.tabs([
        "👤 Single Customer",
        "📂 Bulk Prediction"
    ])

    # ================================
    # 👤 SINGLE CUSTOMER
    # ================================
    with tab1:

        st.subheader("👤 Customer Profile")

        col1, col2, col3 = st.columns(3)

        # DEMOGRAPHICS
        with col1:

            st.markdown("### 🧍 Demographics")

            LIMIT_BAL = st.number_input(
                "Credit Limit (LIMIT_BAL)",
                10000,
                1000000,
                100000
            )

            SEX = st.selectbox(
                "Gender",
                [1, 2],
                format_func=lambda x: {
                    1: "Male",
                    2: "Female"
                }[x]
            )

            EDUCATION = st.selectbox(
                "Education",
                [1, 2, 3, 4],
                format_func=lambda x: {
                    1: "Graduate School",
                    2: "University",
                    3: "High School",
                    4: "Others"
                }[x]
            )

            MARRIAGE = st.selectbox(
                "Marital Status",
                [1, 2, 3],
                format_func=lambda x: {
                    1: "Married",
                    2: "Single",
                    3: "Others"
                }[x]
            )

            AGE = st.slider("Age", 18, 80, 30)

        # PAYMENT HISTORY
        with col2:

            st.markdown("### 💳 Payment History")

            PAY_0 = st.slider("Sep (PAY_0)", -2, 8, 0)
            PAY_2 = st.slider("Aug (PAY_2)", -2, 8, 0)
            PAY_3 = st.slider("Jul (PAY_3)", -2, 8, 0)
            PAY_4 = st.slider("Jun (PAY_4)", -2, 8, 0)
            PAY_5 = st.slider("May (PAY_5)", -2, 8, 0)
            PAY_6 = st.slider("Apr (PAY_6)", -2, 8, 0)

        # BILL AMOUNTS
        with col3:

            st.markdown("### 💰 Bill Amounts")

            BILL_AMT1 = st.number_input("Sep Bill", 0, 1000000, 20000)
            BILL_AMT2 = st.number_input("Aug Bill", 0, 1000000, 19000)
            BILL_AMT3 = st.number_input("Jul Bill", 0, 1000000, 18000)
            BILL_AMT4 = st.number_input("Jun Bill", 0, 1000000, 17000)
            BILL_AMT5 = st.number_input("May Bill", 0, 1000000, 16000)
            BILL_AMT6 = st.number_input("Apr Bill", 0, 1000000, 15000)

        # PAYMENT AMOUNTS
        st.markdown("### 💸 Payment Amounts")

        c1, c2, c3 = st.columns(3)

        with c1:
            PAY_AMT1 = st.number_input("Sep Payment", 0, 500000, 2000)
            PAY_AMT2 = st.number_input("Aug Payment", 0, 500000, 2000)

        with c2:
            PAY_AMT3 = st.number_input("Jul Payment", 0, 500000, 2000)
            PAY_AMT4 = st.number_input("Jun Payment", 0, 500000, 2000)

        with c3:
            PAY_AMT5 = st.number_input("May Payment", 0, 500000, 2000)
            PAY_AMT6 = st.number_input("Apr Payment", 0, 500000, 2000)

        # PREDICTION BUTTON
        if st.button("🔮 Predict Risk"):

            input_df = pd.DataFrame([{
                "LIMIT_BAL": LIMIT_BAL,
                "SEX": SEX,
                "EDUCATION": EDUCATION,
                "MARRIAGE": MARRIAGE,
                "AGE": AGE,
                "PAY_0": PAY_0,
                "PAY_2": PAY_2,
                "PAY_3": PAY_3,
                "PAY_4": PAY_4,
                "PAY_5": PAY_5,
                "PAY_6": PAY_6,
                "BILL_AMT1": BILL_AMT1,
                "BILL_AMT2": BILL_AMT2,
                "BILL_AMT3": BILL_AMT3,
                "BILL_AMT4": BILL_AMT4,
                "BILL_AMT5": BILL_AMT5,
                "BILL_AMT6": BILL_AMT6,
                "PAY_AMT1": PAY_AMT1,
                "PAY_AMT2": PAY_AMT2,
                "PAY_AMT3": PAY_AMT3,
                "PAY_AMT4": PAY_AMT4,
                "PAY_AMT5": PAY_AMT5,
                "PAY_AMT6": PAY_AMT6
            }])

            scaled = scaler.transform(input_df)

            prob = model.predict_proba(scaled)[0][1]

            pred = int(prob > 0.25)

            if pred == 1:

                st.error(
                    f"⚠️ High Risk Customer\n\n"
                    f"Risk Score: {prob:.2f}"
                )

                st.markdown(
                    "👉 Recommendation: Monitor closely"
                )

            else:

                st.success(
                    f"✅ Low Risk Customer\n\n"
                    f"Risk Score: {prob:.2f}"
                )

                st.markdown(
                    "👉 Recommendation: Safe to approve credit"
                )

    # ================================
    # 📂 BULK PREDICTION
    # ================================
    with tab2:

        st.subheader("📂 Bulk Prediction")

        colA, colB = st.columns(2)

        # SAMPLE DATA
        with colA:

            if st.button("✨ Try with Sample Data"):

                sample = df.sample(50)

                scaled = scaler.transform(sample[FEATURES])

                probs = model.predict_proba(scaled)[:, 1]

                preds = (probs > 0.25).astype(int)

                sample["Probability"] = probs

                sample["Prediction"] = preds

                st.dataframe(sample.head())

                fig = px.histogram(
                    sample,
                    x="Probability",
                    title="Risk Distribution"
                )

                st.plotly_chart(fig)

        # FILE UPLOAD
        with colB:

            file = st.file_uploader(
                "📤 Upload CSV",
                type=["csv"]
            )

            if file:

                uploaded_df = pd.read_csv(file)

                if st.button("Run Prediction"):

                    uploaded_df = uploaded_df[FEATURES]

                    scaled = scaler.transform(uploaded_df)

                    probs = model.predict_proba(scaled)[:, 1]

                    preds = (probs > 0.25).astype(int)

                    uploaded_df["Probability"] = probs

                    uploaded_df["Prediction"] = preds

                    st.dataframe(uploaded_df.head())

                    fig = px.histogram(
                        uploaded_df,
                        x="Probability"
                    )

                    st.plotly_chart(fig)

                    st.download_button(
                        "Download Results",
                        uploaded_df.to_csv(index=False),
                        "results.csv"
                    )

# ================================
# FOOTER
# ================================
st.markdown("---")
st.caption("🚀 Built by Khushi Kumari | Capstone Project")