import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import json
from typing import Dict, Any
import os
from auth import AuthManager, render_auth_ui

# Constants
API_URL = "http://localhost:8000"
ALLOWED_EXTENSIONS = ['.csv', '.xlsx']

# Initialize authentication manager
auth_manager = AuthManager(API_URL)

# Page configuration
st.set_page_config(
    page_title="MedSynX - Synthetic Healthcare Data Generation",
    page_icon="🏥",
    layout="wide"
)

# Utility functions
def is_valid_file(filename: str) -> bool:
    return any(filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS)

def display_data_summary(summary: Dict[str, Any]):
    st.subheader("Dataset Summary")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Basic Information:")
        st.write(f"- Number of rows: {summary['shape'][0]}")
        st.write(f"- Number of columns: {summary['shape'][1]}")
        st.write("- Numeric columns:", ", ".join(summary['numeric_columns']))
        st.write("- Categorical columns:", ", ".join(summary['categorical_columns']))
    
    with col2:
        st.write("Missing Values:")
        missing_df = pd.DataFrame.from_dict(summary['missing_values'], orient='index', columns=['Count'])
        st.dataframe(missing_df)

def plot_data_distributions(data: pd.DataFrame):
    st.subheader("Data Distributions")
    
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) > 0:
        selected_col = st.selectbox("Select column for distribution plot:", numeric_cols)
        fig = px.histogram(data, x=selected_col, title=f"Distribution of {selected_col}")
        st.plotly_chart(fig)

def main():
    st.title("🏥 MedSynX")
    st.markdown("### Synthetic Healthcare Data Generation Platform")
    
    # Render authentication UI
    render_auth_ui(auth_manager)
    
    if not auth_manager.is_authenticated():
        st.warning("Please login to use the application.")
        return
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["My Datasets", "Upload Data", "Generate Synthetic Data", "Evaluate Results"])
    
    if page == "My Datasets":
        st.header("My Datasets")
        try:
            response = requests.get(
                f"{API_URL}/api/v1/datasets",
                headers=auth_manager.get_auth_headers()
            )
            
            if response.status_code == 200:
                datasets = response.json()["datasets"]
                if not datasets:
                    st.info("You haven't uploaded any datasets yet.")
                else:
                    for dataset in datasets:
                        with st.expander(f"{dataset['name']} ({dataset['created_at']})"):
                            st.write(f"ID: {dataset['id']}")
                            st.write(f"Type: {'Synthetic' if dataset['is_synthetic'] else 'Original'}")
                            if not dataset['is_synthetic']:
                                if st.button(f"Generate Synthetic Data for {dataset['name']}", key=f"gen_{dataset['id']}"):
                                    st.session_state['selected_dataset'] = dataset
                                    st.session_state['page'] = "Generate Synthetic Data"
                                    st.rerun()
            else:
                st.error("Error fetching datasets")
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    elif page == "Upload Data":
        st.header("Upload Your Dataset")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV or Excel file",
            type=['csv', 'xlsx'],
            help="Upload your healthcare dataset"
        )
        
        if uploaded_file is not None:
            if not is_valid_file(uploaded_file.name):
                st.error("Invalid file type. Please upload a CSV or Excel file.")
                return
                
            try:
                # Upload file to API
                files = {"file": uploaded_file}
                response = requests.post(
                    f"{API_URL}/api/v1/upload",
                    files=files,
                    headers=auth_manager.get_auth_headers()
                )
                
                if response.status_code == 200:
                    result = response.json()
                    st.success("File uploaded successfully!")
                    
                    # Store the dataset ID for later use
                    st.session_state['dataset_id'] = result['dataset_id']
                    
                    # Display data summary
                    display_data_summary(result['data_summary'])
                    
                    # Load and display sample data
                    if uploaded_file.name.endswith('.csv'):
                        data = pd.read_csv(uploaded_file)
                    else:
                        data = pd.read_excel(uploaded_file)
                    
                    st.subheader("Sample Data")
                    st.dataframe(data.head())
                    
                    # Plot distributions
                    plot_data_distributions(data)
                    
                else:
                    st.error(f"Error uploading file: {response.json()['detail']}")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    elif page == "Generate Synthetic Data":
        st.header("Generate Synthetic Data")
        
        # Get dataset selection
        try:
            response = requests.get(
                f"{API_URL}/api/v1/datasets",
                headers=auth_manager.get_auth_headers()
            )
            
            if response.status_code == 200:
                datasets = response.json()["datasets"]
                if not datasets:
                    st.warning("Please upload a dataset first!")
                    return
                    
                # Use selected dataset or let user choose
                if 'selected_dataset' in st.session_state:
                    dataset = st.session_state['selected_dataset']
                    del st.session_state['selected_dataset']
                else:
                    dataset_names = {f"{d['name']} (ID: {d['id']})": d['id'] for d in datasets}
                    selected_name = st.selectbox("Select dataset:", list(dataset_names.keys()))
                    dataset_id = dataset_names[selected_name]
            else:
                st.error("Error fetching datasets")
                return
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
            return
            
        # Model selection
        try:
            response = requests.get(
                f"{API_URL}/api/v1/models",
                headers=auth_manager.get_auth_headers()
            )
            if response.status_code == 200:
                available_models = response.json()['models']
                model_type = st.selectbox("Select synthetic data model:", available_models)
            else:
                st.error("Error fetching available models")
                return
        except Exception as e:
            st.error(f"Error: {str(e)}")
            return
            
        # Privacy parameters
        st.subheader("Privacy Parameters")
        epsilon = st.slider("Epsilon (ε)", 0.1, 10.0, 1.0, 
                          help="Privacy budget - lower values mean more privacy")
        delta = st.number_input("Delta (δ)", 1e-7, 1e-3, 1e-5, format="%.7f",
                              help="Privacy relaxation parameter")
        
        if st.button("Generate Synthetic Data"):
            try:
                # Call generate endpoint
                params = {
                    "dataset_id": dataset_id,
                    "epsilon": epsilon,
                    "delta": delta,
                    "model_type": model_type
                }
                
                with st.spinner("Generating synthetic data..."):
                    response = requests.post(
                        f"{API_URL}/api/v1/generate",
                        json=params,
                        headers=auth_manager.get_auth_headers()
                    )
                    
                if response.status_code == 200:
                    result = response.json()
                    
                    # Store results in session state
                    st.session_state['synthetic_data'] = pd.DataFrame.from_dict(result['synthetic_data'])
                    st.session_state['privacy_metrics'] = result['privacy_metrics']
                    st.session_state['performance_metrics'] = result['performance_metrics']
                    
                    st.success("Synthetic data generated successfully!")
                    
                    # Display sample of synthetic data
                    st.subheader("Sample of Generated Data")
                    st.dataframe(st.session_state['synthetic_data'].head())
                    
                    # Option to download synthetic data
                    csv = st.session_state['synthetic_data'].to_csv(index=False)
                    st.download_button(
                        "Download Synthetic Data",
                        csv,
                        "synthetic_data.csv",
                        "text/csv",
                        key='download-csv'
                    )
                    
                else:
                    st.error(f"Error generating synthetic data: {response.json()['detail']}")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    elif page == "Evaluate Results":
        st.header("Evaluation Results")
        
        if 'synthetic_data' not in st.session_state:
            st.warning("Please generate synthetic data first!")
            return
            
        # Display metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Privacy Metrics")
            st.json(st.session_state['privacy_metrics'])
            
        with col2:
            st.subheader("Performance Metrics")
            st.json(st.session_state['performance_metrics'])
            
        # Visualization comparison
        st.subheader("Data Distribution Comparison")
        if 'synthetic_data' in st.session_state:
            numeric_cols = st.session_state['synthetic_data'].select_dtypes(include=['int64', 'float64']).columns
            if len(numeric_cols) > 0:
                selected_col = st.selectbox("Select column for comparison:", numeric_cols)
                
                fig = px.histogram(st.session_state['synthetic_data'], x=selected_col,
                                 title=f"Distribution Comparison - {selected_col}",
                                 opacity=0.7)
                st.plotly_chart(fig)

if __name__ == "__main__":
    main() 