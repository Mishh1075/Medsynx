import os
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path

def download_synthea():
    """Download Synthea JAR if not present"""
    if not os.path.exists("synthea.jar"):
        print("Downloading Synthea...")
        subprocess.run([
            "curl", "-L", 
            "https://github.com/synthetichealth/synthea/releases/download/v3.1.1/synthea-with-dependencies.jar",
            "-o", "synthea.jar"
        ])

def generate_synthea_data(num_patients=1000, seed=42):
    """Generate synthetic medical data using Synthea"""
    print(f"Generating data for {num_patients} patients...")
    subprocess.run([
        "java", "-jar", "synthea.jar",
        "-p", str(num_patients),
        "--seed", str(seed),
        "--exporter.csv.export", "true",
        "--exporter.json.export", "false",
        "--exporter.fhir.export", "false"
    ])

def preprocess_data():
    """Preprocess Synthea CSV files into clean datasets"""
    # Create output directory
    output_dir = Path("../sample_data")
    output_dir.mkdir(exist_ok=True)
    
    # Read and preprocess patients
    patients = pd.read_csv("output/csv/patients.csv")
    conditions = pd.read_csv("output/csv/conditions.csv")
    medications = pd.read_csv("output/csv/medications.csv")
    observations = pd.read_csv("output/csv/observations.csv")
    
    # Create patient demographics dataset
    demographics = patients[[
        'Id', 'BIRTHDATE', 'DEATHDATE', 'RACE', 'ETHNICITY', 
        'GENDER', 'ZIP'
    ]].copy()
    demographics['AGE'] = (pd.to_datetime('now') - pd.to_datetime(demographics['BIRTHDATE'])).dt.years
    demographics.to_csv(output_dir / "demographics.csv", index=False)
    
    # Create conditions dataset
    conditions_processed = conditions.groupby('PATIENT')['DESCRIPTION'].agg(list).reset_index()
    conditions_processed.columns = ['PATIENT_ID', 'CONDITIONS']
    conditions_processed.to_csv(output_dir / "conditions.csv", index=False)
    
    # Create medications dataset
    medications_processed = medications.groupby('PATIENT')['DESCRIPTION'].agg(list).reset_index()
    medications_processed.columns = ['PATIENT_ID', 'MEDICATIONS']
    medications_processed.to_csv(output_dir / "medications.csv", index=False)
    
    # Create lab results dataset
    lab_results = observations[observations['TYPE'].str.contains('Laboratory', na=False)].copy()
    lab_results = lab_results.pivot_table(
        index='PATIENT',
        columns='DESCRIPTION',
        values='VALUE',
        aggfunc='mean'
    ).reset_index()
    lab_results.to_csv(output_dir / "lab_results.csv", index=False)
    
    # Create vital signs dataset
    vitals = observations[observations['TYPE'].str.contains('Vital Signs', na=False)].copy()
    vitals = vitals.pivot_table(
        index='PATIENT',
        columns='DESCRIPTION',
        values='VALUE',
        aggfunc='mean'
    ).reset_index()
    vitals.to_csv(output_dir / "vital_signs.csv", index=False)
    
    print("Sample datasets created in sample_data/")

def download_chest_xray_samples():
    """Download sample chest X-ray images"""
    image_dir = Path("../sample_data/images")
    image_dir.mkdir(exist_ok=True)
    
    # Download sample images from NIH ChestX-ray14 dataset
    print("Downloading sample chest X-ray images...")
    image_urls = [
        "https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz",
        "https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz"
    ]
    
    for url in image_urls:
        subprocess.run(["curl", "-L", url, "-o", str(image_dir / "images.tar.gz")])
        subprocess.run(["tar", "-xzf", str(image_dir / "images.tar.gz"), "-C", str(image_dir)])
        os.remove(str(image_dir / "images.tar.gz"))

if __name__ == "__main__":
    # Generate Synthea data
    download_synthea()
    generate_synthea_data()
    preprocess_data()
    
    # Download image samples
    download_chest_xray_samples() 