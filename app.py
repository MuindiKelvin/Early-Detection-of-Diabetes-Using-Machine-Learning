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
    initial_sidebar_state="expanded"
)

# Theme configurations
themes = {
    "Medical Blue": {
        "primary": "#0096c7",
        "secondary": "#023e8a",
        "accent": "#48cae4",
        "background": "medical-icons.png"
    },
    "Wellness Green": {
        "primary": "#2d6a4f",
        "secondary": "#40916c",
        "accent": "#95d5b2",
        "background": "wellness-pattern.png"
    },
    "Hospital White": {
        "primary": "#2b2d42",
        "secondary": "#8d99ae",
        "accent": "#edf2f4",
        "background": "hospital-pattern.png"
    },
    "Diagnostic Purple": {
        "primary": "#7209b7",
        "secondary": "#3f37c9",
        "accent": "#4361ee",
        "background": "diagnostic-pattern.png"
    }
}

# Creating Sidebar for theme selection
with st.sidebar:
    st.title("🎨 App Customization")
    selected_theme = st.selectbox(
        "Choose Theme",
        list(themes.keys()),
        index=0
    )
    
    st.markdown("### About This App")
    st.info("""
        This application helps predict diabetes risk based on various health metrics. 
        Select a theme that suits your preference for a more personalized experience.
    """)

# Custom CSS with dynamic theming and sticky header
st.markdown(f"""
    <style>
    /* Sticky Header */
    .main-header {{
        position: sticky;
        top: 0;
        z-index: 999;
        background: linear-gradient(135deg, {themes[selected_theme]["primary"]} 0%, {themes[selected_theme]["secondary"]} 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }}
    
    /* Main background with theme */
    .stApp {{
        background-image: linear-gradient(
            rgba(255, 255, 255, 0.95), 
            rgba(255, 255, 255, 0.95)
        ),
        url("https://www.transparenttextures.com/patterns/{themes[selected_theme]["background"]}");
    }}
    
    /* Card styling */
    .metric-container {{
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        margin: 5px 0;
        border-left: 4px solid {themes[selected_theme]["primary"]};
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }}
    
    /* Input styling */
    .stNumberInput > div > div > input {{
        border-radius: 5px;
        border: 2px solid {themes[selected_theme]["accent"]};
    }}
    
    /* Predict button */
    div.stButton > button:first-child {{
        background: linear-gradient(135deg, {themes[selected_theme]["primary"]} 0%, {themes[selected_theme]["secondary"]} 100%);
        color: white;
        font-size: 18px;
        font-weight: bold;
        width: 100%;
        padding: 0.75em 2em;
        border: none;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }}

    /* Progress bar */
    .progress-bar {{
        height: 25px;
        width: 0%;
        background-color: {themes[selected_theme]["primary"]};
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
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    pregnancies = st.number_input("🤰 Pregnancies", min_value=0, max_value=20, value=0)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    blood_pressure = st.number_input("🩸 Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    glucose = st.number_input("🍬 Glucose Level (mg/dL)", min_value=0, max_value=300, value=120)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    skin_thickness = st.number_input("📏 Skin Thickness (mm)", min_value=0, max_value=100, value=20)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    insulin = st.number_input("💉 Insulin Level (µU/mL)", min_value=0, max_value=900, value=80)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    bmi = st.number_input("📏 BMI (kg/m²)", min_value=0.0, max_value=100.0, value=30.0)
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    dpf = st.number_input("📊 Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    age = st.number_input("🎂 Age (years)", min_value=0, max_value=120, value=25)
    st.markdown('</div>', unsafe_allow_html=True)

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
        time.sleep(0.10)
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

    # Displaying the prediction with theme-colored results
    if prediction[0] == 1:
        st.markdown(f"""
            <div style='background-color: {themes[selected_theme]["accent"]}22; padding: 20px; border-radius: 10px; border-left: 5px solid {themes[selected_theme]["primary"]};'>
                <h3 style='color: {themes[selected_theme]["primary"]}; margin-bottom: 10px;'>🔴 High Risk Detected</h3>
                <p style='font-size: 16px;'>The model indicates an elevated risk of diabetes.</p>
                <p style='font-size: 18px; font-weight: bold; color: {themes[selected_theme]["primary"]};'>
                    Confidence: {prediction_proba[0][1]:.2%}
                </p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div style='background-color: {themes[selected_theme]["accent"]}22; padding: 20px; border-radius: 10px; border-left: 5px solid {themes[selected_theme]["secondary"]};'>
                <h3 style='color: {themes[selected_theme]["secondary"]}; margin-bottom: 10px;'>🟢 Low Risk Detected</h3>
                <p style='font-size: 16px;'>The model indicates a lower risk of diabetes.</p>
                <p style='font-size: 18px; font-weight: bold; color: {themes[selected_theme]["secondary"]};'>
                    Confidence: {prediction_proba[0][0]:.2%}
                </p>
            </div>
        """, unsafe_allow_html=True)

    # Themed disclaimer
    st.markdown(f"""
        <div style='margin-top: 2rem; padding: 1rem; background-color: {themes[selected_theme]["accent"]}11; border-radius: 10px; font-size: 0.9em;'>
            <p><strong>⚠️ Disclaimer:</strong> This tool provides an estimate based on the input data and should not be used as a substitute for professional medical advice. Please consult with a healthcare provider for proper diagnosis and treatment.</p>
        </div>
    """, unsafe_allow_html=True)

# Themed footer
st.markdown(f"""
    <div style='margin-top: 3rem; text-align: center; color: {themes[selected_theme]["primary"]};'>
        <p>💻 Developed by Ashilpa with ❤️ for healthcare | 🏥 Consult your healthcare provider for medical advice</p>
    </div>
""", unsafe_allow_html=True)