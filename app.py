import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS Styling ────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf1 100%);
    }

    /* Title styling */
    .main-title {
        text-align: center;
        padding: 20px 0 5px 0;
        color: #1a1a2e;
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    .sub-title {
        text-align: center;
        color: #666;
        font-size: 1.05rem;
        margin-bottom: 30px;
    }

    /* Prediction result boxes */
    .result-churn {
        background: linear-gradient(135deg, #ff6b6b, #ee0979);
        color: white;
        padding: 25px 30px;
        border-radius: 16px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
        box-shadow: 0 8px 25px rgba(238, 9, 121, 0.35);
        margin: 15px 0;
    }
    .result-no-churn {
        background: linear-gradient(135deg, #56ab2f, #a8e063);
        color: white;
        padding: 25px 30px;
        border-radius: 16px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
        box-shadow: 0 8px 25px rgba(86, 171, 47, 0.35);
        margin: 15px 0;
    }

    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 18px 20px;
        text-align: center;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        margin: 8px 0;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #1a1a2e;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #888;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Section headers */
    .section-header {
        color: #1a1a2e;
        font-size: 1.2rem;
        font-weight: 700;
        border-left: 4px solid #667eea;
        padding-left: 12px;
        margin: 20px 0 15px 0;
    }

    /* Info box */
    .info-box {
        background: #EBF5FB;
        border-left: 4px solid #3498db;
        padding: 14px 18px;
        border-radius: 0 10px 10px 0;
        margin: 10px 0;
        color: #1a5276;
        font-size: 0.95rem;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #aaa;
        font-size: 0.82rem;
        padding: 20px;
        border-top: 1px solid #e0e0e0;
        margin-top: 30px;
    }

    /* Probability bar */
    .prob-bar-container {
        background: #f0f0f0;
        border-radius: 10px;
        overflow: hidden;
        height: 20px;
        margin: 8px 0;
    }
    .prob-bar-fill-high {
        background: linear-gradient(90deg, #ff6b6b, #ee0979);
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s ease;
    }
    .prob-bar-fill-low {
        background: linear-gradient(90deg, #56ab2f, #a8e063);
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s ease;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        color: white;
    }
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label {
        color: #e0e0e0 !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Load Model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

try:
    model = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False


# ── Encoding Maps (must match training encoding) ───────────────────────────────
# LabelEncoder alphabetically sorted these values:
ENCODING = {
    'FrequentFlyer': {
        'No':        0,
        'No Record': 1,
        'Yes':       2
    },
    'AnnualIncomeClass': {
        'High Income':   0,
        'Low Income':    1,
        'Middle Income': 2
    },
    'AccountSyncedToSocialMedia': {
        'No':  0,
        'Yes': 1
    },
    'BookedHotelOrNot': {
        'No':  0,
        'Yes': 1
    }
}


# ── Helper: Preprocess Input ───────────────────────────────────────────────────
def preprocess_input(age, frequent_flyer, annual_income_class,
                     services_opted, account_synced, booked_hotel):
    """Encode user inputs exactly as done during training."""
    input_dict = {
        'Age':                          age,
        'FrequentFlyer':                ENCODING['FrequentFlyer'][frequent_flyer],
        'AnnualIncomeClass':            ENCODING['AnnualIncomeClass'][annual_income_class],
        'ServicesOpted':                services_opted,
        'AccountSyncedToSocialMedia':   ENCODING['AccountSyncedToSocialMedia'][account_synced],
        'BookedHotelOrNot':             ENCODING['BookedHotelOrNot'][booked_hotel]
    }
    return pd.DataFrame([input_dict])


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ✈️ Customer Churn Predictor")
    st.markdown("---")
    st.markdown("### 📌 About This App")
    st.markdown("""
    This app predicts whether a **travel customer is likely to churn** 
    using a trained **Random Forest** model.
    
    Fill in the customer details on the main screen and click 
    **Predict Churn** to get an instant prediction.
    """)
    st.markdown("---")
    st.markdown("### 📊 Model Performance")
    st.markdown("""
    | Metric | Score |
    |--------|-------|
    | Test Accuracy | **89.01%** |
    | ROC AUC | **0.9600** |
    | Algorithm | Random Forest |
    | Trees | 100 |
    | Dataset Size | 954 rows |
    """)
    st.markdown("---")
    st.markdown("### 📋 Feature Guide")
    st.markdown("""
    - **Age**: Customer age (27–38)
    - **Frequent Flyer**: Flyer program status
    - **Income Class**: Annual income bracket
    - **Services Opted**: No. of services used (1–6)
    - **Social Media**: Account linked to social media?
    - **Hotel Booking**: Booked hotel through platform?
    """)
    st.markdown("---")
    st.caption("B.Tech Gen AI · Final Year Project")


# ── Main Page ──────────────────────────────────────────────────────────────────
st.markdown('<h1 class="main-title">✈️ Customer Churn Prediction</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Enter customer details below to predict churn probability using Random Forest</p>', unsafe_allow_html=True)

if not model_loaded:
    st.error("❌ **model.pkl not found!** Please make sure `model.pkl` is in the same directory as `app.py`.")
    st.stop()

# ── Input Form ─────────────────────────────────────────────────────────────────
st.markdown('<p class="section-header">👤 Customer Information</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Demographic Details**")
    age = st.slider(
        "🎂 Age",
        min_value=27, max_value=38, value=32, step=1,
        help="Customer's age (range: 27–38 in dataset)"
    )
    annual_income_class = st.selectbox(
        "💰 Annual Income Class",
        options=['Low Income', 'Middle Income', 'High Income'],
        index=1,
        help="Customer's annual income bracket"
    )

with col2:
    st.markdown("**Travel & Loyalty Details**")
    frequent_flyer = st.selectbox(
        "✈️ Frequent Flyer Status",
        options=['No', 'Yes', 'No Record'],
        index=0,
        help="Is the customer a registered frequent flyer?"
    )
    services_opted = st.slider(
        "🛎️ Number of Services Opted",
        min_value=1, max_value=6, value=3, step=1,
        help="How many travel services has the customer opted for? (1=least, 6=most)"
    )

with col3:
    st.markdown("**Digital Engagement**")
    account_synced = st.selectbox(
        "📱 Account Synced to Social Media",
        options=['No', 'Yes'],
        index=0,
        help="Is the customer's account linked to social media?"
    )
    booked_hotel = st.selectbox(
        "🏨 Booked Hotel or Not",
        options=['No', 'Yes'],
        index=0,
        help="Has the customer booked a hotel through the platform?"
    )

# ── Customer Summary Preview ───────────────────────────────────────────────────
st.markdown("---")
st.markdown('<p class="section-header">📋 Customer Profile Summary</p>', unsafe_allow_html=True)

scol1, scol2, scol3, scol4, scol5, scol6 = st.columns(6)
metrics = [
    (scol1, "Age", str(age), "years"),
    (scol2, "Income", annual_income_class.replace(" Income", ""), "class"),
    (scol3, "Freq. Flyer", frequent_flyer, ""),
    (scol4, "Services", str(services_opted), "opted"),
    (scol5, "Social Media", account_synced, "synced"),
    (scol6, "Hotel Booked", booked_hotel, "")
]
for col, label, value, unit in metrics:
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

# ── Predict Button ─────────────────────────────────────────────────────────────
st.markdown("---")
predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])
with predict_col2:
    predict_button = st.button(
        "🔍 Predict Churn",
        use_container_width=True,
        type="primary"
    )

# ── Prediction Result ──────────────────────────────────────────────────────────
if predict_button:
    # Preprocess input
    input_df = preprocess_input(
        age, frequent_flyer, annual_income_class,
        services_opted, account_synced, booked_hotel
    )

    # Predict
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]
    churn_prob  = probability[1] * 100
    stay_prob   = probability[0] * 100

    st.markdown("---")
    st.markdown('<p class="section-header">🎯 Prediction Result</p>', unsafe_allow_html=True)

    res_col1, res_col2 = st.columns([1, 1])

    with res_col1:
        if prediction == 1:
            st.markdown("""
            <div class="result-churn">
                ⚠️ HIGH CHURN RISK<br>
                <span style="font-size:1rem; font-weight:400;">This customer is likely to churn</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-no-churn">
                ✅ LOW CHURN RISK<br>
                <span style="font-size:1rem; font-weight:400;">This customer is likely to stay</span>
            </div>
            """, unsafe_allow_html=True)

    with res_col2:
        st.markdown("**📊 Prediction Confidence**")

        # Churn probability bar
        st.markdown(f"🔴 Churn Probability: **{churn_prob:.1f}%**")
        fill_class = "prob-bar-fill-high" if churn_prob > 50 else "prob-bar-fill-low"
        st.markdown(f"""
        <div class="prob-bar-container">
            <div class="{fill_class}" style="width:{churn_prob}%"></div>
        </div>
        """, unsafe_allow_html=True)

        # Stay probability bar
        st.markdown(f"🟢 Retention Probability: **{stay_prob:.1f}%**")
        st.markdown(f"""
        <div class="prob-bar-container">
            <div class="prob-bar-fill-low" style="width:{stay_prob}%"></div>
        </div>
        """, unsafe_allow_html=True)

        # Risk level
        st.markdown("**🏷️ Risk Level:**")
        if churn_prob >= 70:
            st.error(f"🔴 CRITICAL RISK — Immediate action needed!")
        elif churn_prob >= 50:
            st.warning(f"🟡 HIGH RISK — Proactive retention recommended")
        elif churn_prob >= 30:
            st.info(f"🔵 MODERATE RISK — Monitor this customer")
        else:
            st.success(f"🟢 LOW RISK — Customer appears loyal")

    # ── Recommendations ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-header">💡 Retention Recommendations</p>', unsafe_allow_html=True)

    rec_col1, rec_col2 = st.columns(2)

    with rec_col1:
        st.markdown("**Based on the prediction:**")
        if prediction == 1:
            recs = []
            if frequent_flyer == 'No' or frequent_flyer == 'No Record':
                recs.append("✈️ Enroll customer in the **Frequent Flyer Loyalty Program** with signup bonus")
            if services_opted <= 2:
                recs.append("🛎️ Offer a **bundled services package** at a discounted rate")
            if account_synced == 'No':
                recs.append("📱 Encourage **Social Media Account Sync** — offer rewards for linking")
            if booked_hotel == 'No':
                recs.append("🏨 Provide a **hotel booking discount coupon** to increase platform engagement")
            if annual_income_class == 'High Income':
                recs.append("💎 Offer **VIP / Premium membership** with exclusive travel perks")
            if not recs:
                recs.append("📞 Schedule a **personalized retention call** with the customer")
                recs.append("🎁 Offer a **surprise loyalty reward** based on past travel history")
            for r in recs:
                st.markdown(f"- {r}")
        else:
            st.markdown("- 🌟 Customer is currently **satisfied** — maintain service quality")
            st.markdown("- 📧 Send a **thank you message** to reinforce loyalty")
            st.markdown("- 🎁 Offer a **loyalty reward** to sustain engagement")
            st.markdown("- 🔔 Keep customer engaged with **relevant travel deals**")

    with rec_col2:
        st.markdown("**Encoded Input Sent to Model:**")
        st.dataframe(input_df, use_container_width=True)
        st.caption("These are the encoded values passed to the Random Forest model.")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Customer Churn Prediction App · Built with Streamlit & Random Forest · B.Tech Gen AI Final Project<br>
    Dataset: Customertravel.csv · Model Accuracy: 89.01% · ROC AUC: 0.9600
</div>
""", unsafe_allow_html=True)
