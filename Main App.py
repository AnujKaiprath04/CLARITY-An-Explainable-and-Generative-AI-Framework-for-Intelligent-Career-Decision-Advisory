"""
Explainable AI-Based Intelligent Life Choice Advisory System
Main Streamlit Application (Interactive Chatbot + Enhanced XAI)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.graph_objects as go
import plotly.express as px
import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
import google.generativeai as genai
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# ==========================================
# PAGE CONFIGURATION & CSS
# ==========================================
st.set_page_config(page_title="CLARITY - AI Career Advisor", page_icon="🧭", layout="wide")

# Modern, user-friendly CSS
st.markdown("""
    <style>
    .main { background-color: #f4f6f9; }
    .stButton>button {
        background-color: #2563eb; color: white; border-radius: 8px; padding: 10px 24px; font-weight: bold; width: 100%; transition: all 0.3s;
    }
    .stButton>button:hover { background-color: #1d4ed8; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    h1, h2, h3 { color: #1e293b; }
    .card {
        background-color: white; padding: 25px; border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 20px; border-left: 5px solid #2563eb;
    }
    .metric-value { font-size: 28px; font-weight: 800; color: #2563eb; }
    .metric-label { font-size: 15px; color: #64748b; font-weight: 600; text-transform: uppercase; }
    .highlight-blue { color: #2563eb; font-weight: bold; }
    .highlight-red { color: #ef4444; font-weight: bold; }
    
    /* Chatbot specific styling */
    .stChatMessage { background-color: white; border-radius: 10px; padding: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); margin-bottom: 10px;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# INITIALIZATION & CACHING
# ==========================================
@st.cache_resource
def load_ml_assets():
    try:
        model = joblib.load('models/career_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        return None, None

model, scaler = load_ml_assets()

CAREER_MAP = {0: "Corporate Job", 1: "Higher Studies", 2: "Startup / Entrepreneurship"}
FINANCE_MAP = {"Low": 0, "Medium": 1, "High": 2}
REVERSE_FINANCE_MAP = {0: "Low", 1: "Medium", 2: "High"}

# Session State Variables
if 'user_data' not in st.session_state: st.session_state['user_data'] = None
if 'prediction' not in st.session_state: st.session_state['prediction'] = None
if 'messages' not in st.session_state: st.session_state['messages'] = []  # Chatbot memory
if 'report_generated' not in st.session_state: st.session_state['report_generated'] = False
if 'pdf_buffer' not in st.session_state: st.session_state['pdf_buffer'] = None

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def generate_pdf_report(user_data, prediction_text, chat_history):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("Intelligent Life Choice Advisory Report", styles['Title']))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("1. User Profile Assessment", styles['Heading2']))
    data = [[k.replace("_", " "), str(v)] for k, v in user_data.items()]
    t = Table(data, colWidths=[200, 100])
    t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                           ('GRID', (0, 0), (-1, -1), 1, colors.black)]))
    elements.append(t)
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("2. ML Model Prediction", styles['Heading2']))
    elements.append(Paragraph(f"Primary Recommendation: <b>{prediction_text}</b>", styles['Normal']))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("3. GenAI Consultation Log", styles['Heading2']))
    for msg in chat_history:
        role = "User: " if msg["role"] == "user" else "AI Advisor: "
        clean_text = msg["content"].replace('**', '')
        elements.append(Paragraph(f"<b>{role}</b> {clean_text}", styles['Normal']))
        elements.append(Spacer(1, 6))

    doc.build(elements)
    buffer.seek(0)
    return buffer

# ==========================================
# UI COMPONENTS
# ==========================================
def assessment_form():
    st.title("Step 1: 📝 Build Your Profile")
    st.markdown("Welcome! Tell the AI about your skills, mindset, and current situation. Be as honest as possible for the best results.")
    
    with st.container():
        st.markdown("### 🧠 Core Capabilities")
        col1, col2 = st.columns(2)
        with col1:
            tech_skill = st.slider("💻 Technical & Core Skills (0-100)", 0, 100, 50)
            comm_skill = st.slider("🗣️ Communication & Soft Skills (0-100)", 0, 100, 50)
        with col2:
            higher_study = st.slider("📚 Interest in Higher Studies (0-100)", 0, 100, 50)
            entrepreneur = st.slider("🚀 Entrepreneurial Mindset (0-100)", 0, 100, 50)

    st.markdown("---")
    
    with st.container():
        st.markdown("### ⚖️ Risk & Stability")
        col3, col4 = st.columns(2)
        with col3:
            risk_tol = st.slider("🎢 Risk Tolerance (0-100)", 0, 100, 50)
        with col4:
            finance = st.selectbox("💰 Current Financial Stability", ["Low", "Medium", "High"])

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔮 Analyze My Profile (Run Machine Learning)", use_container_width=True):
        with st.spinner("Crunching numbers through the Machine Learning model..."):
            st.session_state['user_data'] = {
                'Technical_Skill': tech_skill,
                'Communication_Skill': comm_skill,
                'Financial_Stability': FINANCE_MAP[finance],
                'Risk_Tolerance': risk_tol,
                'Interest_Higher_Studies': higher_study,
                'Entrepreneurial_Mindset': entrepreneur
            }
            
            # ML Prediction Logic
            input_df = pd.DataFrame([st.session_state['user_data']])
            scaled_input = scaler.transform(input_df)
            
            pred = model.predict(scaled_input)[0]
            probs = model.predict_proba(scaled_input)[0]
            
            st.session_state['prediction'] = CAREER_MAP[pred]
            st.session_state['probabilities'] = probs
            st.session_state['scaled_input'] = scaled_input
            
            # Reset Chat if a new assessment is taken
            st.session_state['messages'] = [] 
            
            st.success("✅ Analysis Complete! Please click 'Step 2: Dashboard & XAI' in the sidebar.")

def dashboard_and_xai():
    st.title("Step 2: 📊 Results & Explainable AI (XAI)")
    
    if st.session_state['user_data'] is None:
        st.info("👈 Please complete Step 1 (Assessment Form) first to see your results.")
        return

    st.markdown("### 🎯 AI Primary Recommendation: **<span style='color:#2563eb; font-size: 32px;'>{}</span>**".format(st.session_state['prediction']), unsafe_allow_html=True)

    # Top Metric Cards
    probs = st.session_state['probabilities']
    col1, col2, col3 = st.columns(3)
    col1.markdown(f"<div class='card'><div class='metric-label'>🏢 Corporate Job</div><div class='metric-value'>{probs[0]*100:.1f}%</div></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='card'><div class='metric-label'>🎓 Higher Studies</div><div class='metric-value'>{probs[1]*100:.1f}%</div></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='card'><div class='metric-label'>🚀 Startup / Founder</div><div class='metric-value'>{probs[2]*100:.1f}%</div></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## 🧠 Explainable AI (Why did the AI choose this?)")
    
    with st.expander("❓ What is Explainable AI (XAI)? Click to learn."):
        st.write("""
        Traditional AI is often a **"Black Box"**—it gives you an answer, but doesn't tell you *why*. 
        This system uses advanced **SHAP (SHapley Additive exPlanations)** mathematics from game theory. 
        It breaks open the black box to show exactly which of your personal traits pushed the AI toward its decision, and which traits pushed against it.
        """)

    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(st.session_state['scaled_input'])
    pred_class_idx = list(CAREER_MAP.values()).index(st.session_state['prediction'])
    
    if isinstance(shap_values, list): class_shap_values = shap_values[pred_class_idx][0]
    else: class_shap_values = shap_values[0, :, pred_class_idx]
        
    features = list(st.session_state['user_data'].keys())
    shap_dict = dict(zip(features, class_shap_values))
    sorted_shap = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    st.session_state['top_shap_features'] = str(sorted_shap[:3])

    # Plain English Translation of XAI
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("#### 🗣️ Plain English Translation of the ML Model's logic:")
    for feature, impact in sorted_shap[:3]: 
        friendly_name = feature.replace("_", " ")
        if impact > 0:
            st.markdown(f"- ✅ **<span class='highlight-blue'>Increased Probability:</span>** Your score in **{friendly_name}** strongly convinced the AI to recommend {st.session_state['prediction']}.", unsafe_allow_html=True)
        else:
            st.markdown(f"- ❌ **<span class='highlight-red'>Decreased Probability:</span>** Interestingly, your score in **{friendly_name}** made the AI *hesitate*, but other factors outweighed it.", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Plotly Bar chart for SHAP
    shap_df = pd.DataFrame({'Feature': [f.replace("_", " ") for f in features], 'Impact': class_shap_values})
    shap_df['Color'] = np.where(shap_df['Impact'] > 0, 'Pushed TOWARD recommendation (Positive)', 'Pushed AWAY from recommendation (Negative)')
    
    fig_shap = px.bar(shap_df, x='Impact', y='Feature', orientation='h', 
                      color='Color', color_discrete_map={'Pushed TOWARD recommendation (Positive)': '#2563eb', 'Pushed AWAY from recommendation (Negative)': '#ef4444'})
    fig_shap.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_shap, use_container_width=True)

    st.success("✅ Analysis Understood? Proceed to 'Step 3: GenAI Chatbot' in the sidebar.")

def interactive_chatbot():
    st.title("Step 3: 💬 Interactive AI Mentor (Q&A)")
    
    if st.session_state['prediction'] is None:
        st.info("👈 Please complete Step 1 & 2 first before talking to the mentor.")
        return

    st.markdown("""
    You are now chatting with a **Generative AI Mentor**. It already knows your test scores, your ML recommendation, and your Explainable AI SHAP values. 
    Ask it anything about your career path, risks, or alternative options!
    """)
    
    env_key = os.getenv("GEMINI_API_KEY", "")
    api_key = st.text_input("Google Gemini API Key", value=env_key, type="password", help="Enter your Gemini API key or set GEMINI_API_KEY in .env")

    # Clear chat button & PDF Download
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🗑️ Clear Chat History"):
            st.session_state['messages'] = []
            st.rerun()
    with col2:
        if len(st.session_state['messages']) > 0:
            pdf_buf = generate_pdf_report(st.session_state['user_data'], st.session_state['prediction'], st.session_state['messages'])
            st.download_button("📄 Download Chat Log as PDF Report", data=pdf_buf, file_name="Career_Advisory_Session.pdf", mime="application/pdf")

    st.markdown("---")

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # First time Greeting
    if len(st.session_state.messages) == 0:
        greeting = f"Hello! Based on your profile, our Machine Learning model recommended **{st.session_state['prediction']}**. What specific questions do you have about this path, or would you like a step-by-step action plan to get started?"
        with st.chat_message("assistant"):
            st.markdown(greeting)
        st.session_state.messages.append({"role": "assistant", "content": greeting})

    # Accept user input
    if prompt := st.chat_input("E.g., What are the risks of this path? What masters degree should I do?"):
        if not api_key:
            st.error("Please enter a valid API Key to chat.")
            return
            
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Communicate with Gemini
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            # Build System Persona and Context
            system_instruction = f"""
            You are an empathetic, highly intelligent Explainable AI Career Advisor for an Engineering Student.
            You must use the following contextual data to answer the user's questions:
            [USER DATA]: {st.session_state['user_data']}
            [ML RECOMMENDATION]: {st.session_state['prediction']}
            [TOP SHAP INFLUENCERS (Why ML chose this)]: {st.session_state['top_shap_features']}
            
            Rules:
            1. Keep answers concise, actionable, and structured with bullet points.
            2. Never give generic advice; always tie it back to their specific traits (e.g., "Because your risk tolerance is low...").
            3. Use a professional but encouraging tone.
            """
            
            try:
                genai.configure(api_key=api_key)
                gemini_model = genai.GenerativeModel(
                    model_name='gemini-1.5-flash',
                    system_instruction=system_instruction
                )
                
                # Format Streamlit history to Gemini History format
                gemini_history = []
                for msg in st.session_state.messages[:-1]: # exclude the latest user message
                    role = "user" if msg["role"] == "user" else "model"
                    gemini_history.append({"role": role, "parts": [msg["content"]]})
                
                chat = gemini_model.start_chat(history=gemini_history)
                
                with st.spinner("Analyzing your profile and typing response..."):
                    response = chat.send_message(prompt)
                    full_response = response.text
                    message_placeholder.markdown(full_response)
                
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                error_msg = f"API Error: {str(e)}. Please check your internet connection and API key."
                message_placeholder.error(error_msg)

# ==========================================
# SIDEBAR ROUTING & INFO
# ==========================================
if not model:
    st.error("⚠️ System Offline: Model files not found! Please run `python generate_data.py` and `python train_model.py` in your terminal first.")
    st.stop()

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2042/2042460.png", width=80) 
st.sidebar.title("App Navigation")
st.sidebar.markdown("Follow the steps below:")

menu = st.sidebar.radio("Navigation", ["Step 1: Profile Builder", "Step 2: Dashboard & XAI", "Step 3: GenAI Chatbot"], label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About This Project")
st.sidebar.info("""
**CLARITY: Explainable AI Life Choice Advisor**
Combines Random Forest (ML) for prediction, TreeSHAP (XAI) for transparency, and Google Gemini (GenAI) for personalized, interactive mentoring.
""")

# Route to the correct page
if menu == "Step 1: Profile Builder":
    assessment_form()
elif menu == "Step 2: Dashboard & XAI":
    dashboard_and_xai()
elif menu == "Step 3: GenAI Chatbot":
    interactive_chatbot()