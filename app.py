import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt


# =========================================================
# 1. LOAD MODEL
# =========================================================

model = joblib.load("churn_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# SHAP explainer
explainer = shap.TreeExplainer(model)


# =========================================================
# 2. PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Customer Churn AI",
    page_icon="🤖",
    layout="wide"
)


# =========================================================
# 3. TITLE
# =========================================================

st.title("🤖 Customer Churn Prediction")
st.write(
    "AI/ML powered customer churn prediction and "
    "explainable AI system"
)

st.divider()


# =========================================================
# 4. CUSTOMER INFORMATION
# =========================================================

st.subheader("👤 Customer Information")

col1, col2 = st.columns(2)

with col1:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=72,
        value=12
    )


with col2:

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        max_value=200.0,
        value=70.0
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=840.0
    )


# =========================================================
# 5. ONLINE SERVICES
# =========================================================

st.divider()

st.subheader("🌐 Online Services")

col1, col2, col3 = st.columns(3)

with col1:

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )


with col2:

    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )


with col3:

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )


# =========================================================
# 6. CONTRACT AND PAYMENT
# =========================================================

st.divider()

st.subheader("💳 Contract & Payment")

col1, col2 = st.columns(2)

with col1:

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )


with col2:

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )


# =========================================================
# 7. PREDICTION BUTTON
# =========================================================

st.divider()

predict_button = st.button(
    "🔮 Predict Churn",
    use_container_width=True
)


# =========================================================
# 8. PREDICTION
# =========================================================

if predict_button:

    # -----------------------------------------------------
    # Create customer data
    # -----------------------------------------------------

    customer_data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [senior_citizen],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })


    # -----------------------------------------------------
    # ONE-HOT ENCODING
    # -----------------------------------------------------

    customer_encoded = pd.get_dummies(
        customer_data,
        drop_first=True,
        dtype=int
    )


    # -----------------------------------------------------
    # MATCH TRAINING FEATURES
    # -----------------------------------------------------

    customer_encoded = customer_encoded.reindex(
        columns=feature_columns,
        fill_value=0
    )


    # -----------------------------------------------------
    # MODEL PREDICTION
    # -----------------------------------------------------

    probability = model.predict_proba(
        customer_encoded
    )[0][1]

    prediction = model.predict(
        customer_encoded
    )[0]


    # =====================================================
    # 9. RESULT
    # =====================================================

    st.divider()

    st.subheader("📊 Prediction Result")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Churn Probability",
            f"{probability * 100:.2f}%"
        )

    with col2:

        if probability >= 0.70:
            risk = "High Risk"

        elif probability >= 0.40:
            risk = "Medium Risk"

        else:
            risk = "Low Risk"

        st.metric(
            "Risk Level",
            risk
        )


    # =====================================================
    # 10. PREDICTION MESSAGE
    # =====================================================

    if prediction == 1:

        st.error(
            "🔴 Customer is likely to churn."
        )

    else:

        st.success(
            "🟢 Customer is likely to stay."
        )


    # =====================================================
    # 11. PROBABILITY BAR
    # =====================================================

    st.write("### Churn Probability")

    st.progress(
        float(probability)
    )


    # =====================================================
    # 12. CUSTOMER SUMMARY
    # =====================================================

    st.write("### 👤 Customer Summary")

    summary = pd.DataFrame({
        "Feature": [
            "Tenure",
            "Monthly Charges",
            "Total Charges",
            "Contract",
            "Internet Service",
            "Payment Method"
        ],

        "Value": [
            f"{tenure} months",
            f"{monthly_charges:.2f}",
            f"{total_charges:.2f}",
            contract,
            internet_service,
            payment_method
        ]
    })

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )


    # =====================================================
    # 13. SHAP EXPLAINABLE AI
    # =====================================================

    st.divider()

    st.subheader(
        "🔍 Why did the model make this prediction?"
    )

    try:

        shap_values = explainer(
            customer_encoded
        )

        fig = shap.plots.waterfall(
            shap_values[0],
            show=False
        )

        st.pyplot(
            fig.figure,
            clear_figure=True
        )

        plt.close()

        st.info(
            "SHAP explains which features contributed "
            "to this individual prediction."
        )

    except Exception as e:

        st.warning(
            "SHAP explanation could not be displayed."
        )

        st.write(str(e))