"""
XOXOplanet Exoplanet Detection Interface
NASA Space Apps Challenge 2025

A Streamlit application for detecting exoplanets using machine learning models
trained on NASA datasets.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import joblib
import os
from datetime import datetime
import io

# Set page configuration
st.set_page_config(
    page_title="XOXOplanet - Exoplanet Detection",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "XOXOplanet NASA Space Apps 2025"
    }
)

# Custom CSS for the application - matching Figma design exactly
st.markdown("""
<style>
/* Import Jersey 20 font */
@import url('https://fonts.googleapis.com/css2?family=Jersey+20&display=swap');

body {
    background: #0e0e0e;
    color: white;
    font-family: 'Arial', sans-serif;
}

.stApp {
    background: #0e0e0e;
}

/* Main title styling - exact Figma match */
.main-title {
    font-family: 'Jersey 20', cursive;
    font-size: 64px;
    text-align: center;
    color: #ffffff;
    margin: 2rem 0;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 69px;
}

/* XO circle styling - exact Figma match */
.xo-circle {
    display: inline-block;
    width: 234px;
    height: 226px;
    border-radius: 50%;
    background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
    border: 3px solid #ffffff;
    margin: 0;
    text-align: center;
    line-height: 226px;
    font-size: 200px;
    font-weight: bold;
    color: white;
    box-shadow: 0 0 30px rgba(255, 107, 107, 0.5);
    transition: all 0.3s ease;
    position: relative;
}

.xo-circle:hover {
    transform: scale(1.05);
    box-shadow: 0 0 40px rgba(255, 107, 107, 0.8);
}

/* Animation classes */
.x-disappear {
    opacity: 0;
    transform: scale(0);
    transition: all 0.5s ease;
}

.o-illuminate {
    background: linear-gradient(45deg, #ff0000, #ff6666) !important;
    animation: pulse 1s infinite;
    box-shadow: 0 0 50px rgba(255, 0, 0, 0.8) !important;
}

@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.2); }
    100% { transform: scale(1); }
}

/* Find Out button - exact Figma match */
.find-out-button {
    background: #ba1e68;
    color: white;
    border: 1px solid #404040;
    border-radius: 12px;
    padding: 18px 24px;
    font-size: 24px;
    font-family: 'Jersey 20', cursive;
    font-weight: normal;
    cursor: pointer;
    transition: all 0.3s ease;
    text-transform: none;
    letter-spacing: -0.72px;
    margin: 2rem auto;
    display: block;
    width: 176px;
    height: 63px;
    line-height: 1.5;
}

.find-out-button:hover {
    background: #d42a7a;
    box-shadow: 0 0 20px rgba(186, 30, 104, 0.6);
    transform: translateY(-2px);
}

/* Navigation menu - exact Figma match */
.nav-menu {
    position: fixed;
    bottom: 20px;
    left: 50%;
    transform: translateX(-50%);
    display: flex;
    gap: 16px;
    background: transparent;
    padding: 0;
    border-radius: 0;
    border: none;
    backdrop-filter: none;
    width: 747px;
    height: 84px;
}

.nav-button {
    background: #1a1a1a;
    color: white;
    border: 1px solid #404040;
    border-radius: 8px;
    padding: 24px;
    cursor: pointer;
    transition: all 0.3s ease;
    font-size: 24px;
    font-family: 'Jersey 20', cursive;
    text-transform: none;
    letter-spacing: -0.72px;
    flex: 1;
    height: 84px;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    line-height: 1.5;
}

.nav-button:hover {
    background: #2a2a2a;
    border-color: #606060;
    transform: translateY(-2px);
}

/* Home button - exact Figma match */
.home-button {
    position: fixed;
    top: 18px;
    right: 18px;
    background: transparent;
    color: white;
    border: none;
    border-radius: 0;
    padding: 0;
    cursor: pointer;
    transition: all 0.3s ease;
    font-weight: normal;
    z-index: 1000;
    width: 38px;
    height: 33px;
}

.home-button:hover {
    opacity: 0.8;
}

/* Result text styling */
.result-text {
    text-align: center;
    font-size: 32px;
    font-weight: bold;
    margin: 2rem 0;
    padding: 20px;
    border-radius: 15px;
    background: rgba(0, 0, 0, 0.5);
    border: 1px solid #333;
    font-family: 'Jersey 20', cursive;
}

.exoplanet-found {
    color: #00ff88;
    border-color: #00ff88;
    background: rgba(0, 255, 136, 0.1);
}

.not-exoplanet {
    color: #ff6666;
    border-color: #ff6666;
    background: rgba(255, 102, 102, 0.1);
}

.some-exoplanets {
    color: #ffaa00;
    border-color: #ffaa00;
    background: rgba(255, 170, 0, 0.1);
}

/* Download button */
.download-button {
    background: linear-gradient(45deg, #00aa00, #008800);
    color: white;
    border: 2px solid #ffffff;
    border-radius: 20px;
    padding: 12px 30px;
    font-size: 1.2rem;
    font-weight: bold;
    cursor: pointer;
    transition: all 0.3s ease;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin: 1rem auto;
    display: block;
}

.download-button:hover {
    background: linear-gradient(45deg, #00cc00, #00aa00);
    box-shadow: 0 0 25px rgba(0, 204, 0, 0.6);
    transform: translateY(-2px);
}

/* Hide Streamlit default elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stDeployButton {display:none;}

/* Sidebar styling */
.sidebar .stSelectbox > div > div {
    background: #1a1a1a;
    border: 1px solid #404040;
    color: white;
}

.sidebar .stSlider > div > div {
    background: #1a1a1a;
}

.sidebar .stFileUploader > div {
    background: #1a1a1a;
    border: 1px solid #404040;
}
</style>
""", unsafe_allow_html=True)

def load_trained_models():
    """Load all trained models from the results folder"""
    models = {}
    model_names = ['RandomForest', 'GradientBoosting', 'XGBoost', 'AdaBoost']
    
    for model_name in model_names:
        try:
            model_path = f'results/models/{model_name}.joblib'
            if os.path.exists(model_path):
                models[model_name] = joblib.load(model_path)
            else:
                st.warning(f"Model {model_name} not found at {model_path}")
        except Exception as e:
            st.error(f"Error loading {model_name}: {str(e)}")
    
    return models

def load_model_results():
    """Load model performance results"""
    try:
        with open('results/results.json', 'r') as f:
            return pd.read_json(f)
    except Exception as e:
        st.error(f"Error loading results: {str(e)}")
        return None

def predict_exoplanet(model, data):
    """Make prediction using the selected model"""
    try:
        # Ensure data is in the correct format
        if isinstance(data, pd.DataFrame):
            # For CSV uploads
            predictions = model.predict(data)
            probabilities = model.predict_proba(data)
            return predictions, probabilities
        else:
            # For single row predictions
            data_array = np.array(data).reshape(1, -1)
            prediction = model.predict(data_array)[0]
            probability = model.predict_proba(data_array)[0]
            return prediction, probability
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None

def create_model_comparison_chart(results_df):
    """Create model comparison visualization"""
    fig = go.Figure()
    
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    colors = ['#0066cc', '#00aa88', '#ff6600', '#aa0066', '#6600aa']
    
    for i, metric in enumerate(metrics):
        fig.add_trace(go.Bar(
            name=metric.replace('_', ' ').title(),
            x=results_df['model'],
            y=results_df[metric],
            marker_color=colors[i],
            text=[f'{val:.3f}' for val in results_df[metric]],
            textposition='auto'
        ))
    
    fig.update_layout(
        title="Model Performance Comparison",
        xaxis_title="Models",
        yaxis_title="Score",
        barmode='group',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_roc_curves():
    """Create ROC curves visualization (placeholder)"""
    # This would typically load actual ROC curve data
    # For now, we'll create a sample visualization
    
    fig = go.Figure()
    
    # Sample ROC curve data for each model
    models = ['RandomForest', 'GradientBoosting', 'XGBoost', 'AdaBoost']
    colors = ['#0066cc', '#00aa88', '#ff6600', '#aa0066']
    
    for i, model in enumerate(models):
        # Generate sample ROC curve data
        fpr = np.linspace(0, 1, 100)
        tpr = np.sin(fpr * np.pi / 2) + np.random.normal(0, 0.05, 100)
        tpr = np.clip(tpr, 0, 1)
        
        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=model,
            line=dict(color=colors[i], width=3)
        ))
    
    # Add diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='white', dash='dash', width=2)
    ))
    
    fig.update_layout(
        title="ROC Curves Comparison",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        width=800,
        height=600
    )
    
    return fig

def main_page():
    """Main page with XO planet detection interface - exact Figma design"""
    
    # Home button - exact Figma position
    st.markdown("""
    <div style="position: fixed; top: 18px; right: 18px; z-index: 1000;">
        <button onclick="window.location.href = window.location.href.split('?')[0]" 
                style="background: transparent; border: none; color: white; cursor: pointer; font-size: 24px; width: 38px; height: 33px;">
            🏠
        </button>
    </div>
    """, unsafe_allow_html=True)
    
    # Main title with exact Figma layout - "Is it an" + XO circle + "planet?" + "e"
    st.markdown("""
    <div style="display: flex; align-items: center; justify-content: center; gap: 69px; margin: 2rem 0;">
        <span style="font-family: 'Jersey 20', cursive; font-size: 64px; color: white;">Is it an</span>
        <div style="position: relative; display: inline-block;">
            <div class="xo-circle" id="xo-circle" style="display: flex; align-items: center; justify-content: center;">XO</div>
            <span style="position: absolute; left: -23px; top: -24px; font-family: 'Jersey 20', cursive; font-size: 48px; color: white;">e</span>
        </div>
        <span style="font-family: 'Jersey 20', cursive; font-size: 64px; color: white;">planet?</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Result text area
    result_container = st.empty()
    
    # Find out button - exact Figma positioning (centered under circle)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Find Out", key="find_out_main", help="Analyze the data to determine if it's an exoplanet"):
            # Get current model and data from session state
            if 'selected_model' in st.session_state and 'input_data' in st.session_state:
                models = st.session_state.get('models', {})
                selected_model_name = st.session_state['selected_model']
                
                if selected_model_name in models:
                    model = models[selected_model_name]
                    input_data = st.session_state['input_data']
                    
                    # Make prediction
                    with st.spinner("Analyzing data..."):
                        prediction, probability = predict_exoplanet(model, input_data)
                        
                        if prediction is not None:
                            # Store results
                            st.session_state['last_prediction'] = prediction
                            st.session_state['last_probability'] = probability
                            
                            # Determine result message
                            if isinstance(prediction, np.ndarray):
                                # Multiple predictions (CSV file)
                                exoplanet_count = np.sum(prediction)
                                total_count = len(prediction)
                                
                                if exoplanet_count == 0:
                                    result_text = "It looks like it is not an exoplanet."
                                    result_class = "not-exoplanet"
                                elif exoplanet_count == total_count:
                                    result_text = "You have found an exoplanet!"
                                    result_class = "exoplanet-found"
                                else:
                                    result_text = "It looks like we have some exoplanets here!"
                                    result_class = "some-exoplanets"
                                    
                                # Show result
                                result_container.markdown(f"""
                                <div class="result-text {result_class}">
                                    {result_text}
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Show download button for CSV results
                                if exoplanet_count > 0 and exoplanet_count < total_count:
                                    # Create downloadable CSV
                                    results_df = pd.DataFrame({
                                        'prediction': prediction,
                                        'probability_exoplanet': probability[:, 1] if len(probability.shape) > 1 else [probability[1]],
                                        'probability_not_exoplanet': probability[:, 0] if len(probability.shape) > 1 else [probability[0]]
                                    })
                                    
                                    csv = results_df.to_csv(index=False)
                                    st.download_button(
                                        label="Download Results CSV",
                                        data=csv,
                                        file_name=f"exoplanet_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                        mime="text/csv"
                                    )
                            else:
                                # Single prediction
                                if prediction == 1:
                                    result_text = "You have found an exoplanet!"
                                    result_class = "exoplanet-found"
                                else:
                                    result_text = "It looks like it is not an exoplanet."
                                    result_class = "not-exoplanet"
                                
                                # Show result
                                result_container.markdown(f"""
                                <div class="result-text {result_class}">
                                    {result_text}
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Trigger animation
                            st.markdown("""
                            <script>
                            setTimeout(function() {
                                const xoCircle = document.getElementById('xo-circle');
                                if (xoCircle) {
                                    xoCircle.innerHTML = 'O';
                                    xoCircle.classList.add('o-illuminate');
                                }
                            }, 500);
                            </script>
                            """, unsafe_allow_html=True)
                            
                        else:
                            st.error("Unable to make prediction. Please check your data.")
                else:
                    st.error("Please select a model first.")
            else:
                st.error("Please upload data and select a model first.")
    
    # Navigation menu - using Streamlit buttons instead of JavaScript
    st.markdown("### Navigation")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("About Project", key="nav_about"):
            st.session_state['current_page'] = 'about'
            st.rerun()
    
    with col2:
        if st.button("Upload Data", key="nav_upload"):
            st.session_state['current_page'] = 'upload'
            st.rerun()
    
    with col3:
        if st.button("Select Model", key="nav_model"):
            st.session_state['current_page'] = 'model'
            st.rerun()
    
    with col4:
        if st.button("Analysis", key="nav_analysis"):
            st.session_state['current_page'] = 'analysis'
            st.rerun()

def about_project_page():
    """About Project page"""
    
    # Home button
    if st.button("🏠 Home", key="home_about"):
        st.session_state['current_page'] = 'main'
        st.rerun()
    
    st.markdown("# About Project")
    
    # Try to load about text from file
    try:
        with open('about_project.txt', 'r', encoding='utf-8') as f:
            about_text = f.read()
        st.markdown(about_text)
    except FileNotFoundError:
        st.error("about_project.txt file not found. Please create this file with project information.")
    except Exception as e:
        st.error(f"Error reading about_project.txt: {str(e)}")
    
    # Navigation menu - using Streamlit buttons
    st.markdown("### Navigation")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("About Project", key="nav_about_about"):
            st.session_state['current_page'] = 'about'
            st.rerun()
    
    with col2:
        if st.button("Upload Data", key="nav_upload_about"):
            st.session_state['current_page'] = 'upload'
            st.rerun()
    
    with col3:
        if st.button("Select Model", key="nav_model_about"):
            st.session_state['current_page'] = 'model'
            st.rerun()
    
    with col4:
        if st.button("Analysis", key="nav_analysis_about"):
            st.session_state['current_page'] = 'analysis'
            st.rerun()

def upload_data_sidebar():
    """Upload Data sidebar"""
    
    st.sidebar.markdown("## Upload Data")
    
    # Data input method selection
    input_method = st.sidebar.radio(
            "Choose Input Method:",
            ["Manual Entry (Sliders)", "CSV File Upload"],
            help="Select how to provide exoplanet data for analysis"
        )
    
    if input_method == "Manual Entry (Sliders)":
        st.sidebar.markdown("### Enter Astronomical Parameters")
        
        # Parameter sliders
        orbital_period = st.sidebar.slider(
            "Orbital Period (days)",
            min_value=0.1,
            max_value=1000.0,
            value=25.0,
            step=0.1,
            help="How long it takes the planet to orbit its star"
        )
        
        transit_depth = st.sidebar.slider(
            "Transit Depth (ppm)",
            min_value=0.0,
            max_value=10000.0,
            value=1200.0,
            step=10.0,
            help="Light dimming when planet transits star"
        )
        
        model_snr = st.sidebar.slider(
            "Signal-to-Noise Ratio",
            min_value=0.0,
            max_value=50.0,
            value=8.5,
            step=0.1,
            help="Quality of transit signal"
        )
        
        transit_duration = st.sidebar.slider(
            "Transit Duration (hours)",
            min_value=0.1,
            max_value=24.0,
            value=6.0,
            step=0.1,
            help="How long the transit lasts"
        )
        
        impact_parameter = st.sidebar.slider(
            "Impact Parameter",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.01,
            help="Planet's path across the star"
        )
        
        # Store manual input data
        input_data = [orbital_period, transit_depth, model_snr, transit_duration, impact_parameter]
        st.session_state['input_data'] = input_data
        st.session_state['input_method'] = 'manual'
        
    else:  # CSV File Upload
        st.sidebar.markdown("### Upload CSV File")
        
        uploaded_file = st.sidebar.file_uploader(
            "Choose CSV file", 
            type="csv",
            help="Upload CSV with exoplanet data columns"
        )
        
        if uploaded_file is not None:
            try:
                df_upload = pd.read_csv(uploaded_file)
                st.sidebar.success(f"File uploaded successfully! {len(df_upload)} observations loaded.")
                
                # Display data preview
                st.sidebar.markdown("### Data Preview")
                st.sidebar.dataframe(df_upload.head(), use_container_width=True)
                
                # Store uploaded data
                st.session_state['input_data'] = df_upload
                st.session_state['input_method'] = 'csv'
                st.session_state['uploaded_filename'] = uploaded_file.name
                
            except Exception as e:
                st.sidebar.error(f"Error reading CSV file: {str(e)}")
        else:
            st.sidebar.info("Please upload a CSV file with exoplanet data")

def select_model_sidebar():
    """Select Model sidebar"""
    
    st.sidebar.markdown("## Select Model")
    
    # Load models
    if 'models' not in st.session_state:
        with st.spinner("Loading trained models..."):
            models = load_trained_models()
            st.session_state['models'] = models
    
    models = st.session_state['models']
    
    if models:
        # Model selection dropdown
        model_names = list(models.keys())
        selected_model = st.sidebar.selectbox(
            "Choose Model:",
            model_names,
            help="Select the machine learning model for exoplanet detection"
        )
        
        st.session_state['selected_model'] = selected_model
        
        # Display model info
        st.sidebar.markdown(f"### Selected: {selected_model}")
        
        # Load and display model performance
        results_df = load_model_results()
        if results_df is not None:
            model_results = results_df[results_df['model'] == selected_model]
            if not model_results.empty:
                accuracy = model_results.iloc[0]['accuracy']
                st.sidebar.success(f"Accuracy: {accuracy:.1%}")
                
                # Show other metrics
                precision = model_results.iloc[0]['precision']
                recall = model_results.iloc[0]['recall']
                f1 = model_results.iloc[0]['f1']
                
                st.sidebar.metric("Precision", f"{precision:.1%}")
                st.sidebar.metric("Recall", f"{recall:.1%}")
                st.sidebar.metric("F1-Score", f"{f1:.1%}")
    else:
        st.sidebar.error("No trained models found. Please check the results/models folder.")

def analysis_page():
    """Analysis page with model comparison and ROC curves"""
    
    # Home button
    if st.button("🏠 Home", key="home_analysis"):
        st.session_state['current_page'] = 'main'
        st.rerun()
    
    st.markdown("# Model Analysis")
    
    # Load results
    results_df = load_model_results()
    
    if results_df is not None:
        # Current model accuracy
        if 'selected_model' in st.session_state:
            selected_model = st.session_state['selected_model']
            model_results = results_df[results_df['model'] == selected_model]
            
            if not model_results.empty:
                accuracy = model_results.iloc[0]['accuracy']
                st.markdown(f"## Current Model: {selected_model}")
                st.markdown(f"### Accuracy: {accuracy:.1%}")
        
        # Model comparison chart
        st.markdown("## Model Performance Comparison")
        comparison_fig = create_model_comparison_chart(results_df)
        st.plotly_chart(comparison_fig, use_container_width=True)
        
        # ROC curves
        st.markdown("## ROC Curves")
        roc_fig = create_roc_curves()
        st.plotly_chart(roc_fig, use_container_width=True)
        
        # Detailed metrics table
        st.markdown("## Detailed Performance Metrics")
        st.dataframe(results_df, use_container_width=True)
        
    else:
        st.error("Unable to load model results. Please check the results/results.json file.")
    
    # Navigation menu - using Streamlit buttons
    st.markdown("### Navigation")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("About Project", key="nav_about_analysis"):
            st.session_state['current_page'] = 'about'
            st.rerun()
    
    with col2:
        if st.button("Upload Data", key="nav_upload_analysis"):
            st.session_state['current_page'] = 'upload'
            st.rerun()
    
    with col3:
        if st.button("Select Model", key="nav_model_analysis"):
            st.session_state['current_page'] = 'model'
            st.rerun()
    
    with col4:
        if st.button("Analysis", key="nav_analysis_analysis"):
            st.session_state['current_page'] = 'analysis'
            st.rerun()

def upload_data_page():
    """Upload Data page"""
    
    # Home button
    if st.button("🏠 Home", key="home_upload"):
        st.session_state['current_page'] = 'main'
        st.rerun()
    
    st.markdown("# Upload Data")
    st.info("Use the sidebar on the left to upload your data or enter parameters manually.")
    
    # Show current data status
    if 'input_data' in st.session_state:
        if st.session_state.get('input_method') == 'manual':
            st.success("✅ Manual data entry is active")
            st.write("Current parameters:")
            data = st.session_state['input_data']
            st.write(f"- Orbital Period: {data[0]:.1f} days")
            st.write(f"- Transit Depth: {data[1]:.0f} ppm")
            st.write(f"- Signal-to-Noise Ratio: {data[2]:.1f}")
            st.write(f"- Transit Duration: {data[3]:.1f} hours")
            st.write(f"- Impact Parameter: {data[4]:.2f}")
        else:
            st.success("✅ CSV file uploaded successfully")
            df = st.session_state['input_data']
            st.write(f"**File:** {st.session_state.get('uploaded_filename', 'Unknown')}")
            st.write(f"**Rows:** {len(df)}")
            st.write("**Data Preview:**")
            st.dataframe(df.head(), use_container_width=True)
    else:
        st.warning("⚠️ No data uploaded yet. Please use the sidebar to upload data.")
    
    # Navigation menu - using Streamlit buttons
    st.markdown("### Navigation")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("About Project", key="nav_about_upload"):
            st.session_state['current_page'] = 'about'
            st.rerun()
    
    with col2:
        if st.button("Upload Data", key="nav_upload_upload"):
            st.session_state['current_page'] = 'upload'
            st.rerun()
    
    with col3:
        if st.button("Select Model", key="nav_model_upload"):
            st.session_state['current_page'] = 'model'
            st.rerun()
    
    with col4:
        if st.button("Analysis", key="nav_analysis_upload"):
            st.session_state['current_page'] = 'analysis'
            st.rerun()

def select_model_page():
    """Select Model page"""
    
    # Home button
    if st.button("🏠 Home", key="home_model"):
        st.session_state['current_page'] = 'main'
        st.rerun()
    
    st.markdown("# Select Model")
    st.info("Use the sidebar on the left to select your preferred model.")
    
    # Show current model status
    if 'selected_model' in st.session_state:
        selected_model = st.session_state['selected_model']
        st.success(f"✅ Current model: {selected_model}")
        
        # Load and display model performance
        results_df = load_model_results()
        if results_df is not None:
            model_results = results_df[results_df['model'] == selected_model]
            if not model_results.empty:
                accuracy = model_results.iloc[0]['accuracy']
                precision = model_results.iloc[0]['precision']
                recall = model_results.iloc[0]['recall']
                f1 = model_results.iloc[0]['f1']
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accuracy", f"{accuracy:.1%}")
                with col2:
                    st.metric("Precision", f"{precision:.1%}")
                with col3:
                    st.metric("Recall", f"{recall:.1%}")
                with col4:
                    st.metric("F1-Score", f"{f1:.1%}")
    else:
        st.warning("⚠️ No model selected yet. Please use the sidebar to select a model.")
    
    # Show all available models
    st.markdown("## Available Models")
    models = st.session_state.get('models', {})
    if models:
        for model_name in models.keys():
            st.write(f"- **{model_name}**")
    else:
        st.error("No models loaded. Please check the results/models folder.")
    
    # Navigation menu - using Streamlit buttons
    st.markdown("### Navigation")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("About Project", key="nav_about_model"):
            st.session_state['current_page'] = 'about'
            st.rerun()
    
    with col2:
        if st.button("Upload Data", key="nav_upload_model"):
            st.session_state['current_page'] = 'upload'
            st.rerun()
    
    with col3:
        if st.button("Select Model", key="nav_model_model"):
            st.session_state['current_page'] = 'model'
            st.rerun()
    
    with col4:
        if st.button("Analysis", key="nav_analysis_model"):
            st.session_state['current_page'] = 'analysis'
            st.rerun()

def main():
    """Main application function"""
    
    # Initialize session state
    if 'current_page' not in st.session_state:
        st.session_state['current_page'] = 'main'
    
    # Load models on startup
    if 'models' not in st.session_state:
        with st.spinner("Loading trained models..."):
            models = load_trained_models()
            st.session_state['models'] = models
    
    # Check URL parameters for navigation
    query_params = st.query_params
    if 'page' in query_params:
        st.session_state['current_page'] = query_params['page']
    
    # Sidebar for data upload and model selection (always visible)
    with st.sidebar:
        st.markdown("## Data & Model Settings")
        
        # Data upload and model selection
        upload_data_sidebar()
        st.markdown("---")
        select_model_sidebar()
        
        # Show current status
        st.markdown("---")
        st.markdown("### Current Status")
        
        if 'input_data' in st.session_state:
            if st.session_state.get('input_method') == 'manual':
                st.success("✅ Data: Manual Entry")
            else:
                st.success("✅ Data: CSV Uploaded")
        else:
            st.warning("⚠️ No data uploaded")
            
        if 'selected_model' in st.session_state:
            st.success(f"✅ Model: {st.session_state['selected_model']}")
        else:
            st.warning("⚠️ No model selected")
    
    # Main content area based on current page
    if st.session_state['current_page'] == 'main':
        main_page()
    elif st.session_state['current_page'] == 'about':
        about_project_page()
    elif st.session_state['current_page'] == 'analysis':
        analysis_page()
    elif st.session_state['current_page'] == 'upload':
        upload_data_page()
    elif st.session_state['current_page'] == 'model':
        select_model_page()

if __name__ == "__main__":
    main()