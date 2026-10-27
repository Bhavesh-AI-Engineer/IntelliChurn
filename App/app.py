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


# ============================================================
# PROJECT PATHS
# ============================================================

APP_DIR = Path(__file__).resolve().parent

MODEL_PATH = APP_DIR / "Models" / "intellichurn_model_v5.pkl"

THRESHOLD_PATH = APP_DIR / "Models" / "threshold.txt"


# ============================================================
# LOAD MODEL
# ============================================================

if not MODEL_PATH.exists():

    st.error("❌ IntelliChurn V5 model was not found.")

    st.write(
        f"Expected model location:\n{MODEL_PATH}"
    )

    st.stop()


try:

    model = joblib.load(MODEL_PATH)

except Exception as e:

    st.error("❌ Failed to load IntelliChurn V5 model.")

    st.exception(e)

    st.stop()


# ============================================================
# LOAD THRESHOLD
# ============================================================

THRESHOLD = 0.50


if THRESHOLD_PATH.exists():

    try:

        with open(THRESHOLD_PATH, "r") as f:

            THRESHOLD = float(
                f.read().strip()
            )

    except Exception:

        THRESHOLD = 0.50


# ============================================================
# HEADER
# ============================================================

st.title("📊 IntelliChurn")

st.subheader(
    "Customer Churn Prediction System"
)

st.write(
    "Analyze customer information and identify customers "
    "who may be at risk of churn using the IntelliChurn "
    "V5 machine learning model."
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
        "ℹ️ Satisfaction Score is displayed for customer "
        "context. It is not used by the V5 model."
    )


# ============================================================
# DERIVED CUSTOMER FEATURES
# ============================================================

dependents = (
    "Yes"
    if family_members > 0
    else "No"
)

number_dependents = family_members

under_30 = (
    "Yes"
    if age < 30
    else "No"
)

senior_citizen = (
    "Yes"
    if age >= 65
    else "No"
)


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
    # DERIVED MODEL FEATURES
    # --------------------------------------------------------

    internet_service = (
        "No"
        if internet_type == "No Internet"
        else "Yes"
    )

    phone_service = "Yes"

    multiple_lines = extra_phone_numbers

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
    # FEATURE VALIDATION
    # --------------------------------------------------------

    expected_features = 36

    actual_features = input_data.shape[1]


    if actual_features != expected_features:

        st.error(
            "❌ Feature mismatch detected."
        )

        st.write(
            f"Expected features: {expected_features}"
        )

        st.write(
            f"Received features: {actual_features}"
        )

        st.write(
            "Features being sent to the model:"
        )

        st.write(
            list(input_data.columns)
        )

        st.stop()


    # --------------------------------------------------------
    # SATISFACTION SCORE SAFETY CHECK
    # --------------------------------------------------------

    if "Satisfaction Score" in input_data.columns:

        st.error(
            "❌ Satisfaction Score was accidentally "
            "included in the model input."
        )

        st.stop()


    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    try:

        probabilities = model.predict_proba(
            input_data
        )[0]

    except Exception as e:

        st.error(
            "❌ The IntelliChurn model could not process "
            "the customer information."
        )

        st.exception(e)

        st.stop()


    # --------------------------------------------------------
    # GET CHURN PROBABILITY
    # --------------------------------------------------------

    try:

        classes = list(model.classes_)

    except AttributeError:

        st.error(
            "❌ The loaded model does not expose "
            "class information."
        )

        st.stop()


    if "Yes" in classes:

        churn_index = classes.index("Yes")

    elif 1 in classes:

        churn_index = classes.index(1)

    else:

        st.error(
            "❌ Could not identify the churn class "
            "in the loaded model."
        )

        st.write(
            "Model classes:",
            classes
        )

        st.stop()


    probability = float(
        probabilities[churn_index]
    )

    probability_percent = (
        probability * 100
    )
    estimated_annual_revenue_at_risk = (
        monthly_charge * 12 * probability
    )

    # --------------------------------------------------------
    # FINAL PREDICTION USING V5 THRESHOLD
    # --------------------------------------------------------

    if probability >= THRESHOLD:

        prediction = "Yes"

    else:

        prediction = "No"


    # --------------------------------------------------------
    # RISK CLASSIFICATION
    # --------------------------------------------------------

    if probability >= 0.70:

        risk_level = "High Risk"

    elif probability >= 0.40:

        risk_level = "Medium Risk"

    else:

        risk_level = "Low Risk"


    # ========================================================
    # RESULTS
    # ========================================================

    st.divider()

    st.header("📈 Prediction Result")


    # --------------------------------------------------------
    # MAIN PREDICTION
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
    # METRICS
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
            "Estimated Annual Revenue at Risk",
            f"${estimated_annual_revenue_at_risk:,.2f}"
        )

    st.caption(
        "Revenue estimate = monthly charge × 12 × "
        "churn probability. Assumes the monthly charge "
        "remains constant for 12 months."
    )

    st.caption("Model: IntelliChurn V5")

    # --------------------------------------------------------
    # PROBABILITY PROGRESS BAR
    # --------------------------------------------------------

    st.subheader("Churn Risk")

    st.progress(
        min(
            max(probability, 0.0),
            1.0
        )
    )


    st.caption(
        f"V5 prediction threshold: {THRESHOLD:.2f}"
    )


    # ========================================================
    # BUSINESS RECOMMENDATION
    # ========================================================

    st.subheader(
        "💡 Recommended Action"
    )


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
    # DOWNLOADABLE PREDICTION REPORT
    # ========================================================

    st.subheader("📥 Export Prediction Report")

    report = pd.DataFrame([{
        "Generated At": pd.Timestamp.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "Model Version": "IntelliChurn V5",
        "Prediction": (
            "Likely to CHURN"
            if prediction == "Yes"
            else "Likely to STAY"
        ),
        "Churn Probability (%)": round(
            probability_percent, 2
        ),
        "Risk Level": risk_level,
        "Prediction Threshold": THRESHOLD,
                "Estimated Annual Revenue at Risk ($)": round(
            estimated_annual_revenue_at_risk, 2
        ),
        "Revenue Estimate Basis": (
            "Monthly charge x 12 x churn probability"
        ),
        "Age": age,
        "Gender": gender,
        "Married": married,
        "Family Members": family_members,
        "Satisfaction Score (display only)": satisfaction,
        "Tenure (Months)": tenure,
        "Contract": contract,
        "Monthly Charge ($)": monthly_charge,
        "Payment Method": payment_method,
        "Internet Type": internet_type,
        "Extra Phone Numbers": extra_phone_numbers,
        "Online Security": online_security,
        "Online Backup": online_backup,
        "Device Protection": device_protection,
        "Premium Tech Support": tech_support,
        "Streaming TV": streaming_tv,
        "Streaming Movies": streaming_movies,
        "Streaming Music": streaming_music,
        "Unlimited Data": unlimited_data,
        "Paperless Billing": paperless_billing,
        "Referrals": referrals,
    }])

    report_csv = report.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Prediction Report (CSV)",
        data=report_csv,
        file_name=(
            "intellichurn_prediction_"
            f"{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        ),
        mime="text/csv",
        use_container_width=True,
    )

    st.caption(
        "The report is generated on demand and is not "
        "saved by IntelliChurn to a database."
    )


    # ========================================================
    # CUSTOMER SUMMARY
    # ========================================================

    with st.expander(
        "👤 View Customer Profile"
    ):

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

    with st.expander(
        "🔍 View Model Input"
    ):

        st.dataframe(
            input_data.T.rename(
                columns={0: "Value"}
            ),
            use_container_width=True
        )


        st.caption(
            "Satisfaction Score is intentionally excluded "
            "because the verified V5 model was trained "
            "without it."
        )