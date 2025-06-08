import streamlit as st
import requests
import numpy as np
import io
import plotly.graph_objects as go
from PIL import Image
import json
import pandas as pd

def display_medical_imaging():
    st.title("Medical Image Synthesis")
    st.write("Generate and evaluate synthetic medical images with privacy guarantees")
    
    # Sidebar controls
    st.sidebar.header("Settings")
    epsilon = st.sidebar.slider("Privacy Budget (ε)", 0.1, 10.0, 1.0)
    delta = st.sidebar.slider("Privacy Delta (δ)", 1e-7, 1e-3, 1e-5, format="%.1e")
    
    # Tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["Upload", "Generate", "Evaluate"])
    
    with tab1:
        st.header("Upload Medical Images")
        uploaded_file = st.file_uploader("Choose a medical image file", 
                                       type=["dcm", "nii", "nii.gz", "png", "jpg"])
        
        if uploaded_file is not None:
            try:
                response = requests.post(
                    "http://localhost:8000/api/upload/medical-image",
                    files={"file": uploaded_file},
                    headers={"Authorization": f"Bearer {st.session_state.token}"}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    st.success("Image uploaded successfully!")
                    
                    # Display metadata
                    st.subheader("Image Metadata")
                    st.json(data["metadata"])
                    
                    # Display image info
                    st.subheader("Image Information")
                    st.write(f"Shape: {data['shape']}")
                    
                else:
                    st.error("Failed to upload image")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    with tab2:
        st.header("Generate Synthetic Images")
        num_images = st.number_input("Number of images to generate", 
                                   min_value=1, max_value=10, value=1)
        
        if st.button("Generate"):
            try:
                response = requests.post(
                    "http://localhost:8000/api/generate/medical-image",
                    params={"num_images": num_images,
                           "epsilon": epsilon,
                           "delta": delta},
                    headers={"Authorization": f"Bearer {st.session_state.token}"}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    st.success(f"Generated {data['num_generated']} images!")
                    
                    # Display generated images
                    cols = st.columns(min(data['num_generated'], 3))
                    for idx, img_bytes in enumerate(data['images']):
                        col_idx = idx % 3
                        with cols[col_idx]:
                            img_array = np.load(io.BytesIO(img_bytes))
                            st.image(img_array, caption=f"Synthetic Image {idx+1}")
                            
                else:
                    st.error("Failed to generate images")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    with tab3:
        st.header("Evaluate Synthetic Images")
        
        col1, col2 = st.columns(2)
        with col1:
            real_images = st.file_uploader("Upload real images", 
                                         type=["dcm", "nii", "nii.gz", "png", "jpg"],
                                         accept_multiple_files=True)
        with col2:
            synthetic_images = st.file_uploader("Upload synthetic images", 
                                              type=["dcm", "nii", "nii.gz", "png", "jpg"],
                                              accept_multiple_files=True)
        
        if real_images and synthetic_images and len(real_images) == len(synthetic_images):
            if st.button("Evaluate"):
                try:
                    files = []
                    for real_img, syn_img in zip(real_images, synthetic_images):
                        files.extend([
                            ("real_images", real_img),
                            ("synthetic_images", syn_img)
                        ])
                    
                    response = requests.post(
                        "http://localhost:8000/api/evaluate/medical-image",
                        files=files,
                        headers={"Authorization": f"Bearer {st.session_state.token}"}
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Display metrics
                        st.subheader("Quality Metrics")
                        metrics_df = pd.DataFrame([data['metrics']])
                        st.dataframe(metrics_df)
                        
                        # Display visualizations
                        st.subheader("Visualizations")
                        viz_response = requests.get(
                            f"http://localhost:8000/api/visualization/medical-image/{data['report_path']}",
                            headers={"Authorization": f"Bearer {st.session_state.token}"}
                        )
                        
                        if viz_response.status_code == 200:
                            viz_data = viz_response.json()
                            
                            # Display plots
                            col1, col2 = st.columns(2)
                            with col1:
                                st.image(viz_data['visualization_paths']['metrics_over_time'],
                                        caption="Metrics Over Time")
                            with col2:
                                st.image(viz_data['visualization_paths']['privacy_risks'],
                                        caption="Privacy Risk Assessment")
                            
                            # Display summary
                            st.subheader("Evaluation Summary")
                            st.text(viz_data['summary'])
                            
                    else:
                        st.error("Failed to evaluate images")
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("Please upload an equal number of real and synthetic images") 