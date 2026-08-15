import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="IntelliChurn",
    page_icon="📊",
    layout="wide"
)

# ------------------------------------------------------------
# Project paths
# ------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "Models" / "intellichurn_model_v5.pkl"

# ------------------------------------------------------------
# Load V5 model
# ------------------------------------------------------------

if not MODEL_PATH.exists():
    st.error("❌ IntelliChurn V5 model was not found.")
    st.write(f"Expected location: {MODEL_PATH}")
    st.stop()

model = joblib.load(MODEL_PATH)
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    st.error(
        "❌ IntelliChurn V5 model was not found.\n\n"
        f"Expected location:\n{MODEL_PATH}"
    )
    st.stop()


# ============================================================
# HEADER
# ============================================================

st.title("📊 IntelliChurn")

st.subheader("Customer Churn Prediction System")

st.write(
    "Analyze customer information and identify customers who may be "
    "at risk of churn using the IntelliChurn V5 machine learning model."
)

st.divider()


# ============================================================
# CUSTOMER PROFILE
# ============================================================

st.header("👤 Customer Profile")

col1, col2, col3 = st.columns(3)


with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=35,
        step=1
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    married = st.selectbox(
        "Married",
        ["Yes", "No"]
    )


with col2:

    family_members = st.number_input(
        "Family Members",
        min_value=0,
        max_value=10,
        value=0,
        step=1
    )

    tenure = st.number_input(
        "Tenure (Months)",
        min_value=0,
        max_value=72,
        value=12,
        step=1
    )


with col3:

    satisfaction = st.select_slider(
        "Customer Satisfaction",
        options=[1, 2, 3, 4, 5],
        value=3
    )

    st.caption(
        "ℹ️ Satisfaction Score is displayed for customer context. "
        "It is not used by the V5 model."
    )


# Derived customer information

dependents = "Yes" if family_members > 0 else "No"

number_dependents = family_members

under_30 = "Yes" if age < 30 else "No"

senior_citizen = "Yes" if age >= 65 else "No"


# ============================================================
# CONTRACT & BILLING
# ============================================================

st.header("💳 Contract & Billing")

col1, col2, col3 = st.columns(3)


with col1:

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-Month",
            "One Year",
            "Two Year"
        ]
    )


with col2:

    monthly_charge = st.number_input(
        "Monthly Charge ($)",
        min_value=0.0,
        max_value=200.0,
        value=70.0,
        step=1.0
    )


with col3:

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Bank Withdrawal",
            "Credit Card",
            "Mailed Check"
        ]
    )


# ============================================================
# SERVICES
# ============================================================

st.header("🌐 Services")

col1, col2, col3 = st.columns(3)


with col1:

    internet_type = st.selectbox(
        "Internet Type",
        [
            "Fiber Optic",
            "DSL",
            "Cable",
            "No Internet"
        ]
    )

    extra_phone_numbers = st.selectbox(
        "Extra Phone Numbers",
        [
            "Yes",
            "No"
        ]
    )

    online_security = st.selectbox(
        "Online Security",
        [
            "Yes",
            "No"
        ]
    )

    online_backup = st.selectbox(
        "Online Backup",
        [
            "Yes",
            "No"
        ]
    )


with col2:

    device_protection = st.selectbox(
        "Device Protection",
        [
            "Yes",
            "No"
        ]
    )

    tech_support = st.selectbox(
        "Premium Tech Support",
        [
            "Yes",
            "No"
        ]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        [
            "Yes",
            "No"
        ]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        [
            "Yes",
            "No"
        ]
    )


with col3:

    streaming_music = st.selectbox(
        "Streaming Music",
        [
            "Yes",
            "No"
        ]
    )

    unlimited_data = st.selectbox(
        "Unlimited Data",
        [
            "Yes",
            "No"
        ]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        [
            "Yes",
            "No"
        ]
    )

    referrals = st.number_input(
        "Referrals",
        min_value=0,
        max_value=20,
        value=0,
        step=1
    )


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()

predict_button = st.button(
    "🔮 Predict Churn Risk",
    type="primary",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    # --------------------------------------------------------
    # DERIVED FEATURES
    # --------------------------------------------------------

    # Internet service is automatically determined
    # from the selected internet type.
    internet_service = (
        "No"
        if internet_type == "No Internet"
        else "Yes"
    )

    # Phone service is assumed to be active because this
    # application is designed around existing telecom customers.
    phone_service = "Yes"

    # Multiple Lines is represented in the dataset as Yes/No.
    multiple_lines = extra_phone_numbers

    # --------------------------------------------------------
    # Hidden / Derived Dataset Features
    # --------------------------------------------------------

    # These fields are not shown to the user because they are
    # technical dataset variables rather than useful customer
    # inputs.

    population = 50000

    referred_friend = (
        "Yes"
        if referrals > 0
        else "No"
    )

    offer = "No Offer"

    avg_long_distance = 20.0

    avg_gb_download = 30

    total_charges = (
        monthly_charge * tenure
    )

    total_refunds = 0.0

    total_extra_data = 0.0

    total_long_distance = (
        avg_long_distance * tenure
    )

    total_revenue = (
        total_charges
        + total_extra_data
        + total_long_distance
        - total_refunds
    )

    # CLTV is not entered manually.
    # We estimate it from the customer's revenue and tenure.
    #
    # This keeps the interface simple while providing the model
    # with a value in the same general scale as the training data.

    cltv = (
        total_revenue * 1.25
        + (tenure * 50)
    )


    # --------------------------------------------------------
    # CREATE MODEL INPUT
    # --------------------------------------------------------

    input_data = pd.DataFrame([{

        "Gender": gender,

        "Age": age,

        "Under 30": under_30,

        "Senior Citizen": senior_citizen,

        "Married": married,

        "Dependents": dependents,

        "Number of Dependents": number_dependents,

        "Population": population,

        "Referred a Friend": referred_friend,

        "Number of Referrals": referrals,

        "Tenure in Months": tenure,

        "Offer": offer,

        "Phone Service": phone_service,

        "Avg Monthly Long Distance Charges": avg_long_distance,

        "Multiple Lines": multiple_lines,

        "Internet Service": internet_service,

        "Internet Type": internet_type,

        "Avg Monthly GB Download": avg_gb_download,

        "Online Security": online_security,

        "Online Backup": online_backup,

        "Device Protection Plan": device_protection,

        "Premium Tech Support": tech_support,

        "Streaming TV": streaming_tv,

        "Streaming Movies": streaming_movies,

        "Streaming Music": streaming_music,

        "Unlimited Data": unlimited_data,

        "Contract": contract,

        "Paperless Billing": paperless_billing,

        "Payment Method": payment_method,

        "Monthly Charge": monthly_charge,

        "Total Charges": total_charges,

        "Total Refunds": total_refunds,

        "Total Extra Data Charges": total_extra_data,

        "Total Long Distance Charges": total_long_distance,

        "Total Revenue": total_revenue,

        "CLTV": cltv

    }])


    # --------------------------------------------------------
    # SAFETY CHECK
    # --------------------------------------------------------

    expected_features = 36

    if input_data.shape[1] != expected_features:

        st.error(
            f"❌ Feature mismatch detected.\n\n"
            f"Expected: {expected_features} features\n"
            f"Received: {input_data.shape[1]} features"
        )

        st.stop()


    # Make absolutely sure Satisfaction Score is NOT sent
    if "Satisfaction Score" in input_data.columns:

        st.error(
            "❌ Satisfaction Score was accidentally included "
            "in the model input."
        )

        st.stop()


    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(
        input_data
    )[0][1]

    probability_percent = probability * 100


    # --------------------------------------------------------
    # RISK CLASSIFICATION
    # --------------------------------------------------------

    # V5 threshold = 0.50
THRESHOLD_PATH = APP_DIR / "Models" / "threshold.txt"

if THRESHOLD_PATH.exists():
    with open(THRESHOLD_PATH, "r") as f:
        THRESHOLD = float(f.read().strip())
else:
    THRESHOLD = 0.50


    # ========================================================
    # RESULTS
    # ========================================================

    st.divider()

    st.header("📈 Prediction Result")


    # --------------------------------------------------------
    # Main Prediction
    # --------------------------------------------------------

    if prediction == "Yes":

        st.error(
            "🚨 Customer is likely to CHURN"
        )

    else:

        st.success(
            "✅ Customer is likely to STAY"
        )


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Churn Probability",
            f"{probability_percent:.2f}%"
        )


    with col2:

        st.metric(
            "Risk Level",
            risk_level
        )


    with col3:

        st.metric(
            "Model",
            "IntelliChurn V5"
        )


    # --------------------------------------------------------
    # Probability Progress Bar
    # --------------------------------------------------------

    st.subheader("Churn Risk")

    st.progress(
        min(max(probability, 0.0), 1.0)
    )


    # ========================================================
    # BUSINESS RECOMMENDATION
    # ========================================================

    st.subheader("💡 Recommended Action")


    if risk_level == "High Risk":

        st.error(
            """
            **Immediate retention action recommended.**

            • Contact the customer proactively  
            • Offer a personalized retention incentive  
            • Encourage migration to a longer-term contract  
            • Review pricing and service experience  
            • Consider additional support or service benefits
            """
        )


    elif risk_level == "Medium Risk":

        st.warning(
            """
            **Customer requires monitoring.**

            • Monitor customer engagement  
            • Consider a targeted retention offer  
            • Encourage longer-term contract adoption  
            • Monitor billing and service usage
            """
        )


    else:

        st.success(
            """
            **Customer appears relatively stable.**

            • Continue normal customer engagement  
            • Maintain service quality  
            • Monitor future changes in customer behavior
            """
        )


    # ========================================================
    # CUSTOMER SUMMARY
    # ========================================================

    with st.expander("👤 View Customer Profile"):

        profile_data = pd.DataFrame({

            "Information": [
                "Age",
                "Gender",
                "Married",
                "Family Members",
                "Satisfaction Score",
                "Tenure"
            ],

            "Value": [
                age,
                gender,
                married,
                family_members,
                satisfaction,
                f"{tenure} months"
            ]

        })

        st.dataframe(
            profile_data,
            hide_index=True,
            use_container_width=True
        )


    # ========================================================
    # MODEL INPUT SUMMARY
    # ========================================================

    with st.expander("🔍 View Model Input"):

        st.dataframe(
            input_data.T.rename(
                columns={0: "Value"}
            ),
            use_container_width=True
        )

        st.caption(
            "Satisfaction Score is intentionally excluded because "
            "the verified V5 model was trained without it."
        )