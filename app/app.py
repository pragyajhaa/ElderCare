import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import base64

# Page configuration
st.set_page_config(
    page_title="ElderCare - AI-Powered Health Monitoring",
    page_icon="👵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
def local_css():
    st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
        }
        .stButton>button {
            width: 100%;
            border-radius: 20px;
            border: 1px solid #ff4b4b;
            background-color: #ff4b4b;
            color: white;
            font-weight: bold;
            padding: 0.5rem 1rem;
        }
        .stButton>button:hover {
            background-color: #ff1a1a;
            border-color: #ff1a1a;
        }
        .metric-card {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 1rem;
        }
        .emergency-section {
            background-color: #fff5f5;
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 5px solid #ff4b4b;
            margin: 1rem 0;
        }
    </style>
    """, unsafe_allow_html=True)

# Generate sample ECG data
def generate_ecg_data():
    t = np.linspace(0, 10, 500)
    ecg = np.sin(2 * np.pi * t) + 0.5 * np.sin(2 * np.pi * 2 * t) + 0.2 * np.sin(2 * np.pi * 7 * t)
    return pd.DataFrame({
        'Time': t,
        'ECG': ecg,
        'Timestamp': [datetime.now() - timedelta(seconds=x) for x in reversed(range(500))]
    })

# Generate vitals data
def get_vitals():
    return {
        'temperature': round(np.random.normal(98.6, 0.5), 1),
        'spo2': np.random.randint(95, 100),
        'heart_rate': np.random.randint(60, 100),
        'status': 'Normal'
    }

# Home page
def show_home():
    st.title("👵 ElderCare")
    st.markdown("### AI-Powered Elderly Health Monitoring System")
    
    st.markdown(
        """
        Welcome to ElderCare, an intelligent monitoring system designed to enhance the safety and well-being 
        of elderly individuals through real-time fall detection and health monitoring.
        """
    )
    
    # Navigation instructions
    st.markdown("### Getting Started")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        #### 👉 Live Monitoring
        - Real-time fall detection
        - Health vitals tracking
        - Emergency alerts
        """)
    
    with col2:
        st.markdown("""
        #### 📱 How to Use
        1. Navigate to 'Live Monitoring'
        2. View real-time vitals
        3. Use emergency button if needed
        """)
    
    st.markdown("---")
    st.markdown("### Our Mission")
    st.markdown("""
    At ElderCare, we believe in leveraging cutting-edge AI to provide compassionate care for our elderly loved ones, 
    ensuring their safety and well-being through continuous monitoring and timely interventions.
    """)

# Live Monitoring page
def show_monitoring():
    st.title("📊 Live Monitoring Dashboard")
    
    # Fall detection section
    st.header("Fall Detection")
    st.markdown("Real-time monitoring for fall incidents")
    
    # Placeholder for video feed
    st.video("https://sample-videos.com/video123/mp4/720/big_buck_bunny_720p_1mb.mp4")
    
    # Vitals section
    st.header("Vital Signs")
    
    # Get current vitals
    vitals = get_vitals()
    
    # Create three columns for metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Heart Rate", f"{vitals['heart_rate']} BPM", "Normal")
    with col2:
        st.metric("Temperature", f"{vitals['temperature']}°F", "Normal")
    with col3:
        st.metric("SpO2", f"{vitals['spo2']}%", "Normal")
    
    # ECG Chart
    st.subheader("ECG Monitoring")
    ecg_data = generate_ecg_data()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ecg_data['Time'],
        y=ecg_data['ECG'],
        mode='lines',
        name='ECG',
        line=dict(color='#1f77b4', width=2)
    ))
    
    fig.update_layout(
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        height=300,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Emergency button
    st.markdown("---")
    st.markdown("### Emergency Assistance")
    
    if st.button("🆘 EMERGENCY ALERT", key="emergency_btn"):
        st.session_state.show_emergency_modal = True
    
    if st.session_state.get('show_emergency_modal', False):
        with st.container():
            st.error("🚨 EMERGENCY ALERT ACTIVATED!")
            st.warning("Emergency services have been notified and help is on the way!")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Confirm Emergency"):
                    st.success("✅ Emergency services have been notified!")
                    st.session_state.show_emergency_modal = False
            with col2:
                if st.button("Cancel", type="secondary"):
                    st.session_state.show_emergency_modal = False

# About page
def show_about():
    st.title("ℹ️ About ElderCare")
    
    st.markdown("### Our Vision")
    st.markdown("""
    At ElderCare, we envision a world where technology enables our elderly population to live independently 
    while ensuring their safety and well-being through continuous, non-intrusive monitoring.
    """)
    
    st.markdown("### Meet the Team")
    
    # Team members
    team_cols = st.columns(3)
    
    with team_cols[0]:
        st.image("https://via.placeholder.com/150", width=100)
        st.markdown("""
        **Pragya Jha**  
        *AI/ML Engineer*
        """)

    
    with team_cols[2]:
        st.image("https://via.placeholder.com/150", width=100)
        st.markdown("""
        **Shrey Khandal**  
        *Frontend Developer*
        """)
    
    st.markdown("---")
    st.markdown("### Disclaimer")
    st.warning("""
    This is a demonstration application for educational purposes only. 
    It does not provide medical advice and should not be used as a substitute for professional healthcare.
    """)

# Main app
def main():
    # Initialize session state
    if 'show_emergency_modal' not in st.session_state:
        st.session_state.show_emergency_modal = False
    
    # Apply custom CSS
    local_css()
    
    # Sidebar navigation
    st.sidebar.title("ElderCare")
    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Home", "📊 Live Monitoring", "ℹ️ About"],
        index=0
    )
    
    # Display the selected page
    if page == "🏠 Home":
        show_home()
    elif page == "📊 Live Monitoring":
        show_monitoring()
    elif page == "ℹ️ About":
        show_about()

if __name__ == "__main__":
    main()
