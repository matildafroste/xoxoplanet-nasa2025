"""
XOXOplanet Exoplanet Detection Interface

MODEL INTEGRATION NOTES FOR THE TEAM:
==========================================

This is an application that integrates the ML models.

INTEGRATION POINTS (search for "FUTURE MODEL INTEGRATION POINT"):
1. Line ~108: load_trained_model() function - Replace dummy model with our actual model
2. Line ~308: Single prediction calls - Replace .predict() and .predict_proba()
3. Line ~406: Batch prediction calls - Replace for file uploads



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
import joblib
import json

from src.config import columns_to_remove_KOI_full, columns_to_remove_KOI_subset
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

# Disable the Collapse sidebar button
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Simple dark background CSS
st.markdown("""
<style>
/* Import Jersey 20 font */
@import url('https://fonts.googleapis.com/css2?family=Jersey+20&display=swap');

/* Simple NASA-style dark background */

body {
    background: #000428;
    color: white;
    font-family: 'Jersey 20', cursive;
}

/* Override all text elements to use Jersey font */
* {
    font-family: 'Jersey 20', cursive !important;
}

.stApp {
    background: #000428;
}

.main-header {
    text-align: center;
    color: white;
    font-size: 2rem;
    font-family: 'Jersey 20', cursive;
    margin-bottom: 2rem;
}

.planet-container {
    text-align: center;
    margin: 2rem auto;
}

.sub-header {
    font-family: 'Jersey 20', cursive;
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
    font-family: 'Jersey 20', cursive;
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
    font-family: 'Jersey 20', cursive;
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
    font-family: 'Jersey 20', cursive;
}

.exoplanet-not-detected {
    background: #333333;
    border: 1px solid #555555;
    padding: 1rem;
    color: white;
    text-align: center;
    font-size: 1.2rem;
    font-family: 'Jersey 20', cursive;
}

.stButton > button {
    background: #0066cc;
    color: white;
    border: 1px solid #004499;
    padding: 0.75rem 2rem;
    font-size: 1rem;
    font-family: 'Jersey 20', cursive;
    font-weight: 400;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    transition: background 0.3s ease;
}

.stButton > button:hover {
    background: #004499;
    border-color: #003366;
}

/* XO circle styling - exact Figma match from app_test2.py */
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

.menu-section {
    background: rgba(0, 0, 0, 0.8);
    padding: 1.5rem;
    border: 1px solid #333333;
    margin: 1rem 0;
}

.menu-title {
    font-family: 'Jersey 20', cursive;
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
    """
    NOTE: This function is not used in the current application.
    Loads all datasets.
    """
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

def load_chosen_data(data_choice = ""):
    """
    Loads the chosen dataset and returns it as a pandas dataframe.
    Parameters:
        data_choice (string): "Kepler full" or "Kepler subset"

    Returns:
        dataframe: The corresponding dataset

    Raises:
        NameError: when any other string is given
    """

    if data_choice == "Kepler full":
        path = "datasets/KOI_2025.10.03_07.23.34.csv"
        skiprow_nr = 144
    elif data_choice == "Kepler subset": 
        path = 'datasets/KOI_cumulative.csv'
        skiprow_nr = 53
    else:
        print(NameError)

    df_orig = pd.read_csv(path, skiprows=skiprow_nr)
    return df_orig

def load_trained_model():
    """
    NOTE: This function is not used in the current applicaton.
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
    """
    NOTE: This function is not used in the current application.
    Create dramatic planet transformation animation
    """
    
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

def load_results_models(results_path="results/models"):
    """
    Load all saved joblib models from a folder into a dictionary.

    Parameters:
        results_path (str): Path where models are saved.

    Returns:
        dict: {model_name: trained_model}
    """
    models = {}
    for file in os.listdir(results_path):
        if file.endswith(".joblib"):
            model_name = file.replace(".joblib", "")
            models[model_name] = joblib.load(os.path.join(results_path, file))
            print(f"✅ Loaded model: {model_name}")
    return models

def load_results_info(results_path="results/"):
    """
    Load saved results from results.json.

    Parameters:
        results_path (str): Path where results.json is saved.

    Returns:
        dict: Loaded results dictionary
    """
    file_path = os.path.join(results_path, "results.json")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No results.json found in {results_path}")
    
    with open(file_path, "r") as f:
        results = json.load(f)
    
    print(f"Loaded results from {file_path}")
    return results

def restructure_results(res_list):
    results_dict = {}

    for val in res_list:
        model_name = val["model"]  # grab model name
        # take all other keys except "model"
        scores = {k: v for k, v in val.items() if k != "model"}  
        results_dict[model_name] = scores

    return results_dict

def get_data_info(df, data_choice):
    df_in = df.copy()

    if data_choice == "Kepler full":
        columns_to_remove = columns_to_remove_KOI_full.copy()
    elif data_choice == "Kepler subset":   # fixed typo: "fubset" -> "subset"
        columns_to_remove = columns_to_remove_KOI_subset.copy()
    else:
        raise NameError(f"Invalid data_choice: {data_choice}")
    # all columns
    all_columns = df_in.columns
    # always remove label column
    columns_to_remove.append("koi_disposition")

    non_numeric_columns = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]
    columns_to_remove.extend(non_numeric_columns)

    # use .difference() to remove unwanted columns
    feature_columns = all_columns.difference(columns_to_remove)
    data_info_dict = {
        "All columns": list(all_columns),
        "Feature columns": list(feature_columns)
    }
    return data_info_dict


def show_model_scores(scores: dict, model_name: str = "Model Performance"):
    """
    Display model performance scores in a nice Streamlit table.
    
    Parameters:
        scores (dict): Dictionary with metrics like accuracy, precision, recall, f1, roc_auc.
        model_name (str): Optional name of the model.
    """
    # Convert to DataFrame for clean display
    df_scores = pd.DataFrame(scores, index=[model_name]).T
    df_scores.columns = ["Score"]

    # Round for readability
    df_scores["Score"] = df_scores["Score"].round(4)

    st.markdown(f"{model_name} Scores")
    st.table(df_scores)



def main():
    """Main application function"""

    # Header
    st.markdown('<h1 class="main-header">XOXOPLANET DETECTION SYSTEM</h1>', unsafe_allow_html=True)
    
    # Sidebar with navigation menu. This is where the inputs and model used is specified.
    with st.sidebar:
        # Select whether to use the Kepler dataset with all variables or a just subset (default)
        st.markdown("## DATA SELECTION")
        data_choice = st.selectbox(
            "Select Data Source:", 
            ["Kepler subset", "Kepler full"],
            index=0,
            help="Choose which Kepler data you want you model to have been trained on. Full is around 100 features, subset is trained on 40 features. You will see all features further down in the side bar."
        )

        data_df = load_chosen_data(data_choice)
        # Update session state
        st.session_state['data_name'] = data_choice
        st.session_state['data'] = data_df

        st.markdown("## MODEL SELECTION")
        
        # Model Selection Dropdown (expandable list)
        models_dict = load_results_models(results_path=f"results/{data_choice}/models/")
        results_list = load_results_info(results_path=f"results/{data_choice}/") 
        
        results_dict = restructure_results(results_list)
        
        available_models = results_dict.keys()
        
        model_choice = st.selectbox(
            "Select Model:", 
            available_models,
            index=0,
            help="Choose the machine learning model for exoplanet detection"
        )

        # Update session state
        st.session_state['model'] = models_dict[model_choice]
        st.session_state['model_info'] = results_dict[model_choice]
        
        # Table where the model's performance metrics are shown
        show_model_scores(scores = results_dict[model_choice])
        
        # Get the relevant dataset features
        data_info = get_data_info(data_df, data_choice) 
        features = data_info["Feature columns"]

        # Input Section
        st.markdown("## INPUT DATA")
        
        # Data Input Choice (Manual vs CSV vs Random from dataset)
        input_method = st.radio(
            "Choose Input Method:",
            ["Manual Entry", "CSV File Upload", "Existing input data from test set"],
            help="Select how to provide exoplanet data for analysis. If choosing existing you will get a randomized potential canditate form the test dataset."
        )
        st.markdown(input_method)
        if input_method == "Manual Entry": # Let the user manually input own data
            names_and_descr = json.load(open('datasets/variable_names_and_descriptions.json'))
            # TODO : get the most important features?
            st.markdown("Enter astronomical observation data:")
            input_values =  {}
            for feature in features:
                if feature in names_and_descr:
                    var_name = names_and_descr[feature]['name']
                    var_descr = names_and_descr[feature]['description']
                else:
                    var_name = feature
                    var_descr = 'Could not find description.'

                # Take manual input as strings, thus allowing empty fields.
                # Display an error message below the input field if not a number.
                input_text = st.text_input(label=var_name,
                                           help=var_descr,
                                           key=feature)
                if input_text != '':
                    try:
                        input_text = float(input_text)  # try to convert the variable to a float
                    except ValueError:  # if it's not convertible, catch the ValueError exception
                        st.error(f"Invalid input, only numbers are allowed.")
                input_values[feature] = input_text
                
            # Load feature order 
            with open(f"results/{data_choice}/features.json", "r") as f: 
                features_from_model_in_order = json.load(f) 
                # Create one-row DataFrame with NaNs in correct feature order 
            input_df = pd.DataFrame([np.nan] * len(
                features_from_model_in_order), index=features_from_model_in_order
                ).T

            empty_fields_ratio = 0 # Create a variable to keep track of how many fields are left empty
            # Update with provided input values 
            for key, value in input_values.items(): 
                if value=='': # If the variable's input field is left empty
                    empty_fields_ratio += 1/len(input_values)
                    value = data_df[key].mean() # Impute using the average value of the dataset
                if key in input_df.columns: 
                    input_df.at[0, key] = value # Now input_df is ready for prediction 
            # Load scaler 
            scaler = joblib.load(f"results/{data_choice}/scaler.pkl") 
            # Scale your input before predicting 
            input_df_scaled = scaler.transform(input_df) 
            input_to_predict = input_df_scaled
                
                
        elif input_method == "CSV File Upload":  # CSV File Upload
            st.info("Upload CSV file with exoplanet data:")
            
            uploaded_file = st.file_uploader(
                "Choose CSV file", 
                type="csv",
                help="Upload CSV with columns as specified by Kepler Objects of Interest: koi_disposition, ..."
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
        elif input_method == "Existing input data from test set":
            file_path = os.path.join(f"results/{data_choice}/", "X_test_scaled.npy")
            arr = np.load(file_path)

            # pick random row
            random_idx = np.random.choice(arr.shape[0])
            input_to_predict = arr[random_idx].reshape(1, -1)

        # Add link to KOI data
        st.markdown("**[About Nasa Data](%s)**" % "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative")
    
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
        
        st.markdown("""
        <div class="planet-container" style="text-align: center; margin: 2rem 0;">
            <div class="xo-circle" id="xo-circle" style="display: flex; align-items: center; justify-content: center; margin: 0 auto;">XO</div>
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
            font-family: 'Jersey 20', cursive !important;
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
        
        col1, col2, col3 = st.columns([1, 3, 1])

        with col2:
            if st.button("IS IT AN EXOPLANET?", key="analyze_button", 
                       help="Click to analyze the mysterious object",
                       use_container_width=True):

                if input_method == "Manual Entry" and empty_fields_ratio>=0.5:
                    st.error('Warning: Fewer than half of the available variables have been selected. Model reliability may be significantly reduced, and the resulting likelihood estimates should be interpreted with caution. Consider including more observational parameters for a more robust analysis.')
                
                # Analyze with trained model
                with st.spinner("Analyzing mysterious object..."):
                    model = st.session_state['model']
                    
                    # Create input data for the trained model features
                    # st.markdown(input_to_predict)
                    prediction = model.predict(input_to_predict)[0]
                    confidence = model.predict_proba(input_to_predict)[0][1]
      
                    # Store result
                    st.session_state['last_result'] = prediction
                    st.session_state['last_confidence'] = confidence
                    
                    # Show detailed result
                    if prediction == 1:
                        st.success(f"**EXOPLANET DETECTED!**")
                        st.info(f"**This object appears to be an exoplanet with confidence level {confidence:.1%}**")
                    else:
                        st.error(f"**CANDIDATE DETECTED**")
                        st.info(f"This object does not with certainty appear to be an exoplanet - this is classified as a Canditate with confidence level {confidence:.1%}.")
                    
                    # Trigger XO circle animation - change to O like test app
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
        
        
        # Description below planet
        st.markdown("""
        <div style="text-align: center; margin-top: 3rem;">
            <p style="color: #CCCCCC; font-size: 1.2rem; font-family: 'Jersey 20', cursive;">
                Machine learning framework for exoplanet classification using the Kepler Objects of Interest database
            </p>
            <p style="color: #999999; font-size: 1rem; font-family: 'Jersey 20', cursive; margin-top: 1rem;">
                Modify the input variables or model selection in the sidebar to try different scenarios
            </p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()