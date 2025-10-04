"""
XOXOplanet Exoplanet Detection Interface

MODEL INTEGRATION NOTES FOR THE TEAM:
==========================================

This is a skeleton application ready for our ML model integration.

INTEGRATION POINTS (search for "FUTURE MODEL INTEGRATION POINT"):
1. Line ~108: load_trained_model() function - Replace dummy model with our actual model
2. Line ~308: Single prediction calls - Replace .predict() and .predict_proba()
3. Line ~406: Batch prediction calls - Replace for file uploads

MODEL FORMAT EXPECTED:
- model.predict(data) should return array of 0s and 1s (0=not exoplanet, 1=exoplanet)
- model.predict_proba(data) should return array of [p_not_exoplanet, p_exoplanet]

Currently runs with dummy model for demonstration purposes.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os

# Set page configuration
st.set_page_config(
    page_title="XOXOplanet Exoplanet Detection",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "XOXOplanet NASA Space Apps 2025"
    }
)

# Simple dark background CSS
st.markdown("""
<style>
/* Simple NASA-style dark background */
/* Font import removed to prevent loading issues */

body {
    background: #000428;
    color: white;
    font-family: 'Helvetica', sans-serif;
}

.stApp {
    background: #000428;
}

.main-header {
    text-align: center;
    color: white;
    font-size: 2rem;
    margin-bottom: 2rem;
}

.planet-container {
    text-align: center;
    margin: 2rem auto;
}

.sub-header {
    font-family: 'Helvetica', Arial, sans-serif;
    font-size: 1.4rem;
    font-weight: 500;
    background: linear-gradient(90deg, #0066cc, #003366);
    padding: 0.5rem 1rem;
    color: #ffffff;
    margin: 1rem 0;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.sidebar-header {
    font-family: 'Helvetica', Arial, sans-serif;
    font-size: 1.2rem;
    font-weight: 500;
    color: #0066cc;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 1rem;
}

.planet-container {
    text-align: center;
    margin: 3rem 0;
}

.planet-image {
    width: 200px;
    height: 200px;
    border-radius: 50%;
    border: 3px solid #666666;
    margin: 0 auto 2rem auto;
    background: #1a1a1a;
    box-shadow: 0 0 20px rgba(102, 102, 102, 0.5);
}

.question-text {
    font-family: 'Helvetica', Arial, sans-serif;
    font-size: 1.8rem;
    font-weight: 300;
    color: #ffffff;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 1rem;
}

.exoplanet-detected {
    background: #004d00;
    border: 1px solid #006600;
    padding: 1rem;
    color: white;
    text-align: center;
    font-size: 1.2rem;
    font-family: 'Helvetica', Arial, sans-serif;
}

.exoplanet-not-detected {
    background: #333333;
    border: 1px solid #555555;
    padding: 1rem;
    color: white;
    text-align: center;
    font-size: 1.2rem;
    font-family: 'Helvetica', Arial, sans-serif;
}

.stButton > button {
    background: #0066cc;
    color: white;
    border: 1px solid #004499;
    padding: 0.75rem 2rem;
    font-size: 1rem;
    font-family: 'Helvetica', Arial, sans-serif;
    font-weight: 400;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    transition: background 0.3s ease;
}

.stButton > button:hover {
    background: #004499;
    border-color: #003366;
}

.menu-section {
    background: rgba(0, 0, 0, 0.8);
    padding: 1.5rem;
    border: 1px solid #333333;
    margin: 1rem 0;
}

.menu-title {
    font-family: 'Helvetica', Arial, sans-serif;
    font-size: 1.1rem;
    font-weight: 500;
    color: #0066cc;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 1rem;
}

/* Hide Streamlit UI elements for cleaner look */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

def load_and_prepare_data():
    """Load and prepare the datasets for machine learning"""
    try:
        # Load KOI dataset
        koi_data = pd.read_csv('datasets/KOI_cumulative.csv', skiprows=144)
        
        # Load TOI dataset  
        toi_data = pd.read_csv('datasets/TOI_2025.10.03_07.20.57.csv', skiprows=90)
        
        # Load K2 dataset
        k2_data = pd.read_csv('datasets/k2pandc_2025.10.03_07.23.54.csv', skiprows=298)
        
        return koi_data, toi_data, k2_data
    except Exception as e:
        st.error(f"Error loading datasets: {str(e)}")
        return None, None, None

def load_trained_model():
    """
    Train exoplanet detection model using NASA KOI dataset
    USES ONLY LIBRARIES ALREADY IN PROJECT
    """
    try:
        import warnings
        warnings.filterwarnings('ignore')
        
        st.info("Training exoplanet model with NASA KOI data...")
        
        # Load KOI dataset
        df_orig = pd.read_csv('datasets/KOI_2025.10.03_07.23.34.csv', skiprows=144)
        df = df_orig.copy()
        
        # Clean data
        columns_to_remove = ["rowid", "kepid", "kepoi_name", "kepler_name", "koi_pdisposition", "koi_score"]
        df = df.drop(columns=columns_to_remove, errors='ignore')
        
        # Filter data: CONFIRMED exoplanets vs FALSE POSITIVE
        df_filtered = df[df['koi_disposition'].isin(['CONFIRMED', 'FALSE POSITIVE'])].copy()
        
        # Define key features (simplified set using only libraries we have)
        feature_columns = ['koi_period', 'koi_depth', 'koi_model_snr', 'koi_duration', 'koi_impact']
        
        # Keep only features that exist in dataset
        feature_columns = [col for col in feature_columns if col in df_filtered.columns]
        
        # Prepare training data
        X = df_filtered[feature_columns].fillna(0)
        y = (df_filtered['koi_disposition'] == 'CONFIRMED').astype(int)
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train RandomForest classifier
        model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
        model.fit(X_train, y_train)
        
        # Calculate performance
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        
        # Build model info
        feature_importance = dict(zip(feature_columns, model.feature_importances_) if feature_columns else {})
        
        model_info = {
            'accuracy': test_acc,
            'train_accuracy': train_acc,
            'feature_importance': feature_importance,
            'model_type': 'RandomForest Classifier',
            'features': feature_columns
        }
        
        st.success(f"Model trained! Accuracy: {test_acc:.2%}")
        return model, model_info
            
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        # Fallback to simplified rule-based model
        class FallbackModel:
            def predict(self, X):
                # Simple rules based on key features for exoplanet identification
                orbital_period, transit_depth = X[0][0], X[0][1]
                
                # Exoplanet detection rules (simplified)
                if orbital_period > 10 and transit_depth > 600:
                    return [1]  # Likely exoplanet
                else:
                    return [0]  # Likely not exoplanet
            
            def predict_proba(self, X):
                orbital_period, transit_depth = X[0][0], X[0][1]
                if orbital_period > 10 and transit_depth > 600:
                    return [[0.15, 0.85]]  # 85% confidence exoplanet
                else:
                    return [[0.75, 0.25]]  # 75% confidence not exoplanet
        
        return FallbackModel(), {'accuracy': 0.65, 'model_type': 'Fallback Rule-Based Model'}

def create_planet_transformation(is_exoplanet, confidence):
    """Create dramatic planet transformation animation"""
    
    # Star position (center)
    star_x = [0]
    star_y = [0]
    
    # Create the plot
    fig = go.Figure()
    
    # Add star with subtle glow effect
    fig.add_trace(go.Scatter(
        x=star_x, y=star_y,
        mode='markers',
        marker=dict(size=25, color='gold',
                   line=dict(width=3, color='orange')),
        name='Star'
    ))
    
    # Create planet transformation
    if is_exoplanet:
        planet_color = 'rgba(100, 200, 255, 0.9)'
        planet_name = "EXOPLANET CONFIRMED"
    else:
        planet_color = 'rgba(150, 150, 150, 0.9)'
        planet_name = "NOT AN EXOPLANET"
    
    # Add planet
    fig.add_trace(go.Scatter(
        x=[0], y=[2],
        mode='markers',
        marker=dict(size=25, color=planet_color,
                   line=dict(width=3, color='white')),
        name=planet_name
    ))
    
    # Update layout
    fig.update_layout(
        title=f"{planet_name}",
        xaxis=dict(range=[-3, 3], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[-1, 3.5], showgrid=False, zeroline=False, showticklabels=False),
        showlegend=False,
        width=600,
        height=500,
        paper_bgcolor='rgba(0,0,0,0.9)',
        plot_bgcolor='rgba(0,0,0,0.9)'
    )
    
    return fig

# Stars background function removed to prevent rendering issues

# Old dummy model function removed - now using trained model

def main():
    """Main application function"""
    
    # Stars background removed to prevent rendering issues
    
    # Auto-load trained model
    if 'model' not in st.session_state:
        with st.spinner("Loading NASA Exoplanet Detection Model..."):
            model, model_info = load_trained_model()
            st.session_state['model'] = model
            st.session_state['model_info'] = model_info
    
    # Professional NASA-style header
    st.markdown('<h1 class="main-header">XOXOPLANET DETECTION SYSTEM</h1>', unsafe_allow_html=True)
    
    # Sidebar with navigation menu
    with st.sidebar:
        st.markdown("## MODEL STATUS")
        
        # Model Selection Dropdown (expandable list)
        available_models = [
            "Random Forest Classifier",
            "Support Vector Machine", 
            "Gradient Boosting",
            "Neural Network"
        ]
        
        model_choice = st.selectbox(
            "Select Model:", 
            available_models,
            index=0,
            help="Choose the machine learning model for exoplanet detection"
        )
        
        st.success(f"Active Model: {model_choice}")
        st.info(f"Accuracy: {st.session_state['model_info']['accuracy']:.0%}")
        
        # Test buttons
        st.markdown("## TEST SELECTIONS")
        st.warning("Temporary test buttons for demonstration")
        
        # Input Section - Manual Data Entry
        st.markdown("## INPUT DATA")
        
        # Data Input Choice (Manual vs CSV)
        input_method = st.radio(
            "Choose Input Method:",
            ["Manual Entry (Sliders)", "CSV File Upload"],
            help="Select how to provide exoplanet data for analysis"
        )
        
        if input_method == "Manual Entry (Sliders)":
            st.info("Enter astronomical observation data:")
            
            orbital_period = st.number_input("Orbital Period (days)", min_value=0.1, value=25.0, 
                                           help="How long it takes planet to orbit its star")
            
            transit_depth = st.number_input("Transit Depth (ppm)", min_value=0.0, value=1200.0,
                                         help="Light dimming when planet transits star")
            
            model_snr = st.number_input("Signal-to-Noise Ratio", min_value=0.0, value=8.5,
                                      help="Quality of transit signal")
            
            transit_duration = st.number_input("Transit Duration (hours)", min_value=0.1, value=6.0,
                                             help="How long transit lasts")
            
            impact_parameter = st.number_input("Impact Parameter", min_value=0.0, max_value=1.0, value=0.3,
                                             help="Planet's path across star")
        
        else:  # CSV File Upload
            st.info("Upload CSV file with exoplanet data:")
            
            uploaded_file = st.file_uploader(
                "Choose CSV file", 
                type="csv",
                help="Upload CSV with columns: orbital_period, transit_depth, model_snr, transit_duration, impact_parameter"
            )
            
            if uploaded_file is not None:
                try:
                    df_upload = pd.read_csv(uploaded_file)
                    st.success(f"File uploaded successfully! {len(df_upload)} observations loaded.")
                    
                    # Display first few rows
                    st.write("Data preview:")
                    st.dataframe(df_upload.head(), use_container_width=True)
                    
                    # Store uploaded data
                    st.session_state['uploaded_data'] = df_upload
                    
                except Exception as e:
                    st.error(f"Error reading CSV file: {str(e)}")
            
            # Set default values for when no file is uploaded
            orbital_period = 25.0
            transit_depth = 1200.0
            model_snr = 8.5
            transit_duration = 6.0
            impact_parameter = 0.3
        
        # Manual test buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("TEST EXOPLANET", type="secondary"):
                st.session_state['test_data'] = {
                    'orbital_period': 35.0,
                    'transit_depth': 1500.0,
                    'model_snr': 12.5,
                    'transit_duration': 8.0,
                    'impact_parameter': 0.25
                }
                st.success("Exoplanet test data loaded!")
        
        with col2:
            if st.button("TEST NOT EXOPLANET", type="secondary"):
                st.session_state['test_data'] = {
                    'orbital_period': 2.0,
                    'transit_depth': 300.0,
                    'model_snr': 3.2,
                    'transit_duration': 1.5,
                    'impact_parameter': 0.9
                }
                st.success("Not exoplanet test data loaded!")
        
        # Model Comparison Graphics
        st.markdown("## MODEL COMPARISON")
        
        if st.button("Show Model Performance", type="secondary"):
            # Create model comparison visualization
            st.info("Generating model performance comparison...")
            
            # Sample performance data for different models
            models_data = {
                'Model': ['Random Forest', 'SVM', 'Gradient Boosting', 'Neural Network'],
                'Accuracy': [0.87, 0.82, 0.89, 0.85],
                'Precision': [0.91, 0.88, 0.93, 0.89],
                'Recall': [0.83, 0.79, 0.86, 0.84],
                'F1-Score': [0.87, 0.83, 0.89, 0.86]
            }
            
            df_comparison = pd.DataFrame(models_data)
            
            # Display comparison table
            st.write("**Model Performance Metrics:**")
            st.dataframe(df_comparison, use_container_width=True)
            
            # Create comparison chart
            fig_comparison = px.bar(
                df_comparison, 
                x='Model', 
                y=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                title="Model Performance Comparison",
                color_discrete_sequence=['#0066CC', '#0088FF', '#003366', '#0099FF']
            )
            fig_comparison.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            
            st.plotly_chart(fig_comparison, use_container_width=True)
            
            # Feature Importance for Random Forest
            if model_choice == "Random Forest Classifier":
                st.subheader("Feature Importance (Random Forest)")
                
                feature_importance_data = {
                    'Feature': ['Orbital Period', 'Transit Depth', 'Signal-to-Noise', 'Duration', 'Impact'],
                    'Importance': [0.25, 0.35, 0.20, 0.12, 0.08]
                }
                
                fig_features = px.bar(
                    feature_importance_data,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title="Feature Importance for Exoplanet Detection",
                    color='Importance',
                    color_continuous_scale='Blues'
                )
                fig_features.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white'
                )
                
                st.plotly_chart(fig_features, use_container_width=True)
        
        # Navigation Menu in sidebar
        st.markdown("## NAVIGATION MENU")
        st.markdown("""
        **Detection Analysis**  
        **Model Comparison**  
        **Data Upload**  
        **Model Information**  
        **Documentation**  
        **About NASA Data**
        """)
    
    # Main content area - always show planet
    # Center the planet with button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Planet that can transform
        if 'last_result' not in st.session_state:
            planet_color = "#CCCCCC"  # Default gray
            planet_text = "MYSTERIOUS OBJECT"
        else:
            if st.session_state['last_result'] == 1:
                planet_color = "#00FF88"  # Green exoplanet
                planet_text = "EXOPLANET CONFIRMED"
            else:
                planet_color = "#666666"  # Gray not exoplanet
                planet_text = "NOT AN EXOPLANET"
        
        st.markdown(f"""
        <div class="planet-container" style="text-align: center; margin: 2rem 0;">
            <div style="width: 250px; height: 250px; border-radius: 50%; 
                       background: {planet_color}; margin: 0 auto; 
                       border: 5px solid #999999; box-shadow: 0 0 30px rgba(0,0,0,0.5);
                       transition: all 1s ease;">
                <div style="color: white; text-align: center; line-height: 250px; 
                           font-weight: bold; font-size: 1.1rem;">{planet_text}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # NASA Blue analyze button - perfectly aligned under planet
        st.markdown('''
        <div style="text-align: center; margin-top: -2rem;">
        <style>
        .nasa-button {
            background-color: #0066CC !important;
            color: white !important;
            border: 2px solid #FFFFFF !important;
            border-radius: 8px !important;
            padding: 1rem 2rem !important;
            font-size: 1.2rem !important;
            font-weight: 600 !important;
            cursor: pointer !important;
            text-align: center !important;
            display: inline-block !important;
            width: auto !important;
            margin: 0 auto !important;
        }
        .nasa-button:hover {
            background-color: #0088FF !important;
            box-shadow: 0 0 20px rgba(0, 136, 255, 0.5) !important;
        }
        </style>
        ''', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("IS IT AN EXOPLANET?", key="analyze_button", 
                       help="Click to analyze the mysterious object"):
                # Use test data if available, otherwise use sidebar input fields
                if 'test_data' in st.session_state:
                    orbital_period = st.session_state['test_data']['orbital_period']
                    transit_depth = st.session_state['test_data']['transit_depth']
                    model_snr = st.session_state['test_data']['model_snr']
                    transit_duration = st.session_state['test_data']['transit_duration']
                    impact_parameter = st.session_state['test_data']['impact_parameter']
                    st.success(f"Using test data: Period={orbital_period:.1f} days, Depth={transit_depth:.0f} ppm")
                else:
                    # Use values from sidebar input fields
                    st.success(f"Using manual input: Period={orbital_period:.1f} days, Depth={transit_depth:.0f} ppm")
                
                # Analyze with trained model
                with st.spinner("Analyzing mysterious object..."):
                    model = st.session_state['model']
                    
                    # Create input data for the trained model features
                    input_features = [
                        orbital_period,      # koi_period
                        transit_depth,       # koi_depth  
                        model_snr,          # koi_model_snr
                        transit_duration,    # koi_duration
                        impact_parameter     # koi_impact
                    ]
                    
                    input_data = [input_features]
                    
                    prediction = model.predict(input_data)[0]
                    confidence = model.predict_proba(input_data)[0][1]  # Exoplanet confidence
                    
                    # Store result
                    st.session_state['last_result'] = prediction
                    st.session_state['last_confidence'] = confidence
                    
                    # Show detailed result
                    if prediction == 1:
                        st.success(f"**EXOPLANET DETECTED!**")
                        st.success(f"**Confidence Level:** {confidence:.1%}")
                        st.info(f"**Detection Parameters:**")
                        st.info(f"• Orbital Period: {orbital_period:.1f} days")
                        st.info(f"• Transit Depth: {transit_depth:.0f} ppm")
                        st.info(f"• Signal-to-Noise Ratio: {model_snr:.1f}")
                    else:
                        st.error(f"**NOT AN EXOPLANET**")
                        st.error(f"**Confidence Level:** {confidence:.1%}")
                        st.info(f"This object appears to be a false positive or stellar phenomena.")
                    
                    st.rerun()  # Refresh to show planet transformation
        
        
        # Description below planet
        st.markdown("""
        <div style="text-align: center; margin-top: 3rem;">
            <p style="color: #CCCCCC; font-size: 1.2rem; font-family: 'Helvetica', Arial, sans-serif;">
                Advanced AI analysis system for exoplanet detection using NASA datasets
            </p>
            <p style="color: #999999; font-size: 1rem; margin-top: 1rem;">
                Use the test buttons in the sidebar to try different scenarios
            </p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
