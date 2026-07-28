#Importing Necessary Libraries for the project Application and Model integration
import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
import time

# Setting Streamlit page configuration
st.set_page_config(
    page_title="Diabetes Risk Prediction App",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Fixed color palette (previously the "Medical Blue" theme option).
# The theme picker was removed to keep the app simple — one clean look
# instead of a customization menu in the sidebar.
PRIMARY = "#0096c7"
SECONDARY = "#023e8a"
ACCENT = "#48cae4"

# A single dark, readable text color used for all body copy sitting on light cards,
# regardless of whether the visitor's browser/OS is in light or dark mode.
BODY_TEXT_COLOR = "#1b1b1f"
MUTED_TEXT_COLOR = "#3a3a3f"

# The sidebar is fully removed (see CSS below) — no controls, no content.
# The "About This App" note now lives in the main content area instead,
# right above the input fields.

# Custom CSS
st.markdown(f"""
    <style>
    /* Force a light color scheme for the whole app so custom light-colored
       cards never collide with a dark-mode browser's default light text. */
    :root {{
        color-scheme: light;
    }}

    /* --- Typography: a cleaner, more modern typeface than the browser default --- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, .stApp {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }}

    /* --- Fluid layout: keeps content readable and centered on large
       monitors, and removes wasted/cramped padding on small screens. --- */
    [data-testid="stAppViewContainer"] .main .block-container {{
        max-width: 1200px;
        margin: 0 auto;
        padding-left: clamp(1rem, 4vw, 3rem);
        padding-right: clamp(1rem, 4vw, 3rem);
        padding-top: clamp(1rem, 3vw, 2rem);
    }}

    * {{
        box-sizing: border-box;
    }}

    /* --- Responsive input grid ---
       Streamlit's 4 equal columns get cramped on tablets and overflow on
       phones. Reflow to 2-per-row on medium screens and 1-per-row on
       narrow ones, instead of relying on Streamlit's default behavior. */
    @media (max-width: 1024px) {{
        [data-testid="stHorizontalBlock"] > [data-testid="column"] {{
            flex: 1 1 50% !important;
            min-width: 240px !important;
        }}
    }}
    @media (max-width: 640px) {{
        [data-testid="stHorizontalBlock"] > [data-testid="column"] {{
            flex: 1 1 100% !important;
            min-width: 100% !important;
        }}
    }}

    /* Make sure any un-styled text defaults to a readable dark color */
    .stApp, .stApp p, .stApp span, .stApp div {{
        color: {BODY_TEXT_COLOR};
    }}

    /* --- Sidebar: removed completely, per request. Hides both the
       sidebar panel itself and the small arrow control used to reopen it,
       so no trace of it remains in the layout. --- */
    [data-testid="stSidebar"] {{
        display: none !important;
    }}
    [data-testid="collapsedControl"] {{
        display: none !important;
    }}

    /* About card — sits in the main content area, above the inputs. */
    .about-card {{
        background-color: {ACCENT}1a;
        border-left: 4px solid {PRIMARY};
        border-radius: 10px;
        padding: clamp(1rem, 3vw, 1.5rem);
        margin-bottom: clamp(1.25rem, 4vw, 2rem);
    }}
    .about-card h3 {{
        color: {PRIMARY};
        margin: 0 0 0.5rem 0;
        font-size: clamp(1.05rem, 2.8vw, 1.25rem);
    }}
    .about-card p {{
        color: {BODY_TEXT_COLOR};
        margin: 0;
        font-size: clamp(0.9rem, 2.2vw, 1rem);
        line-height: 1.5;
    }}

    /* Sticky Header — fluid type sizing so it never overflows on phones */
    .main-header {{
        position: sticky;
        top: 0;
        z-index: 999;
        background: linear-gradient(135deg, {PRIMARY} 0%, {SECONDARY} 100%);
        padding: clamp(1rem, 3vw, 1.5rem);
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: clamp(1.25rem, 4vw, 2rem);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }}
    .main-header, .main-header * {{
        color: white;
    }}
    .main-header h1 {{
        font-size: clamp(1.35rem, 4.2vw, 2.1rem);
        line-height: 1.25;
        margin: 0;
    }}
    .main-header p {{
        font-size: clamp(0.9rem, 2.2vw, 1.2rem) !important;
        opacity: 0.92;
    }}
    .main-header::after {{
        content: "";
        display: block;
        width: 56px;
        height: 4px;
        background: rgba(255, 255, 255, 0.7);
        border-radius: 999px;
        margin: 0.9rem auto 0;
    }}

    /* Main app background */
    .stApp {{
        background-color: whitesmoke;
    }}

    /* Card styling — applied directly to each number input's own container,
       since wrapping widgets in separate open/close st.markdown() divs
       doesn't actually nest them in Streamlit (it produces a disconnected,
       empty div rendered next to the widget instead of around it). */
    [data-testid="stNumberInput"] {{
        background-color: white;
        padding: 10px 15px;
        border-radius: 8px;
        margin: 5px 0 15px 0;
        border-left: 4px solid {PRIMARY};
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        transition: box-shadow 0.2s ease, transform 0.2s ease;
    }}
    [data-testid="stNumberInput"]:focus-within {{
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        transform: translateY(-1px);
    }}

    /* Input styling - covers the input box AND its wrapper/stepper buttons,
       so nothing falls back to a dark-mode widget skin. */
    .stNumberInput input {{
        border-radius: 5px;
        border: 2px solid {ACCENT};
        color: {BODY_TEXT_COLOR} !important;
        background-color: #ffffff !important;
    }}
    [data-testid="stNumberInput"] > div,
    [data-testid="stNumberInputStepUp"],
    [data-testid="stNumberInputStepDown"] {{
        background-color: #ffffff !important;
        color: {BODY_TEXT_COLOR} !important;
        border-color: {ACCENT} !important;
    }}

    /* Predict button */
    div.stButton > button:first-child {{
        background: linear-gradient(135deg, {PRIMARY} 0%, {SECONDARY} 100%);
        color: white;
        font-size: clamp(0.95rem, 2.5vw, 18px);
        font-weight: bold;
        width: 100%;
        padding: 0.75em 2em;
        border: none;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.2s ease;
    }}
    div.stButton > button:first-child:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 14px rgba(0, 0, 0, 0.18);
    }}
    div.stButton > button:first-child:active {{
        transform: translateY(0);
    }}
    div.stButton > button:first-child * {{
        color: white;
    }}

    /* Progress bar */
    .progress-bar {{
        height: 25px;
        width: 0%;
        background-color: {PRIMARY};
        text-align: right;
        line-height: 25px;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        transition: width 0.5s ease-in-out;
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0 10px;
    }}
    .progress-bar * {{
        color: white;
    }}

    /* Result and disclaimer cards always get explicit dark text,
       independent of the accent color used for their tinted background. */
    .result-card, .result-card p {{
        color: {BODY_TEXT_COLOR};
    }}
    .result-card {{
        padding: clamp(1rem, 3vw, 20px) !important;
    }}
    .result-card h3 {{
        font-size: clamp(1.1rem, 3vw, 1.4rem) !important;
    }}
    .disclaimer-card, .disclaimer-card p, .disclaimer-card strong {{
        color: {MUTED_TEXT_COLOR};
    }}
    .disclaimer-card {{
        padding: clamp(0.75rem, 2.5vw, 1rem) !important;
    }}

    /* Footer: stacks and centers gracefully on narrow screens instead of
       forcing everything onto one long line that gets clipped. */
    .app-footer {{
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        align-items: center;
        gap: 0.4rem 0.8rem;
        text-align: center;
        font-size: clamp(0.8rem, 2vw, 1rem);
    }}

    /* Fade out animation */
    @keyframes fadeout {{
        from {{
            opacity: 1;
        }}
        to {{
            opacity: 0;
        }}
    }}
    </style>
""", unsafe_allow_html=True)

# App header
st.markdown("""
    <div class="main-header">
        <h1>🩺 Diabetes Risk Prediction App</h1>
        <p style='font-size: 1.2em; margin-top: 1rem;'>
            Enter your health metrics below to assess your diabetes risk
        </p>
    </div>
""", unsafe_allow_html=True)

# About card — replaces the old sidebar note, now shown inline above the inputs
st.markdown("""
    <div class="about-card">
        <h3>ℹ️ About This App</h3>
        <p>This application helps predict diabetes risk based on various health
        metrics. Enter your values below and click <strong>Predict Risk</strong>.</p>
    </div>
""", unsafe_allow_html=True)

# Loading the already trained model & pre-trained scaler
try:
    model = joblib.load('knn_best_diabetes_model.pkl')
    scaler = joblib.load('scaler.pkl')  # Loading the pre-trained scaler
except:
    st.error("⚠️ Error loading the model. Please ensure the model file is present in the app directory.")
    st.stop()

# Creating the app appearance to be in four columns for input fields
col1, col2, col3, col4 = st.columns(4)

# Distributing inputs across four columns
with col1:
    pregnancies = st.number_input("🤰 Pregnancies", min_value=0, max_value=20, value=0)
    blood_pressure = st.number_input("🩸 Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70)

with col2:
    glucose = st.number_input("🍬 Glucose Level (mg/dL)", min_value=0, max_value=300, value=120)
    skin_thickness = st.number_input("📏 Skin Thickness (mm)", min_value=0, max_value=100, value=20)

with col3:
    insulin = st.number_input("💉 Insulin Level (µU/mL)", min_value=0, max_value=900, value=80)
    bmi = st.number_input("📏 BMI (kg/m²)", min_value=0.0, max_value=100.0, value=30.0)

with col4:
    dpf = st.number_input("📊 Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
    age = st.number_input("🎂 Age (years)", min_value=0, max_value=120, value=25)

# Collecting the targeted user input
user_input = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])

# Scaling the collected user input
user_input_scaled = scaler.transform(user_input)  # Applying the pre-trained scaler to user input

# Center the predict button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button("Predict Risk 🔍")

if predict_button:
    # Showing the loading progress bar
    progress_bar = st.markdown("<div class='progress-bar' style='width: 0%;'><span class='progress-text'>0%</span><span class='analysis-message'>⏳ Gathering health data...</span></div>", unsafe_allow_html=True)

    # Simulating the required analysis process
    for percentage in range(0, 101):
        time.sleep(0.02)
        analysis_message = "⏳ Loading user health data..."
        if percentage < 25:
            analysis_message = "⏳ Gathering health data..."
        elif percentage < 50:
            analysis_message = "🧮 Performing calculations..."
        elif percentage < 75:
            analysis_message = "📊 Analyzing risk factors..."
        else:
            analysis_message = "🔍 Finalizing prediction..."
        progress_bar.write(f"<div class='progress-bar' style='width: {percentage}%;'><span class='progress-text'>{percentage}%</span><span class='analysis-message'>{analysis_message}</span></div>", unsafe_allow_html=True)

    with st.spinner('Analyzing your health metrics...'):
        prediction = model.predict(user_input_scaled)
        prediction_proba = model.predict_proba(user_input_scaled)

    # After the results are available, let's fade out the progress bar
    progress_bar.write("""
        <style>
        .progress-bar {
            animation: fadeout 1s ease-out;
            animation-fill-mode: forwards;
        }
        </style>
        <div class='progress-bar' style='width: 100%;'><span class='progress-text'>100%</span><span class='analysis-message'>🔍 Finalizing prediction...</span></div>
    """, unsafe_allow_html=True)

    # Displaying the prediction with theme-colored results.
    # Every text element below has an explicit color so it can never
    # inherit an invisible light color from a dark-mode browser.
    if prediction[0] == 1:
        st.markdown(f"""
            <div class="result-card" style='background-color: {ACCENT}22; padding: 20px; border-radius: 10px; border-left: 5px solid {PRIMARY};'>
                <h3 style='color: {PRIMARY}; margin-bottom: 10px;'>🔴 High Risk Detected</h3>
                <p style='font-size: 16px; color: {BODY_TEXT_COLOR};'>The model indicates an elevated risk of diabetes.</p>
                <p style='font-size: 18px; font-weight: bold; color: {PRIMARY};'>
                    Confidence: {prediction_proba[0][1]:.2%}
                </p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="result-card" style='background-color: {ACCENT}22; padding: 20px; border-radius: 10px; border-left: 5px solid {SECONDARY};'>
                <h3 style='color: {SECONDARY}; margin-bottom: 10px;'>🟢 Low Risk Detected</h3>
                <p style='font-size: 16px; color: {BODY_TEXT_COLOR};'>The model indicates a lower risk of diabetes.</p>
                <p style='font-size: 18px; font-weight: bold; color: {SECONDARY};'>
                    Confidence: {prediction_proba[0][0]:.2%}
                </p>
            </div>
        """, unsafe_allow_html=True)

    # Themed disclaimer
    st.markdown(f"""
        <div class="disclaimer-card" style='margin-top: 2rem; padding: 1rem; background-color: {ACCENT}11; border-radius: 10px; font-size: 0.9em;'>
            <p style='color: {MUTED_TEXT_COLOR};'><strong style='color: {MUTED_TEXT_COLOR};'>⚠️ Disclaimer:</strong> This tool provides an estimate based on the input data and should not be used as a substitute for professional medical advice. Please consult with a healthcare provider for proper diagnosis and treatment.</p>
        </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown(f"""
    <div class="app-footer" style='margin-top: 3rem; color: {PRIMARY};'>
        <span>💻 Developed by Muindi with ❤️ for healthcare</span>
        <span aria-hidden="true">|</span>
        <span>🏥 Consult your healthcare provider for medical advice</span>
    </div>
""", unsafe_allow_html=True)
