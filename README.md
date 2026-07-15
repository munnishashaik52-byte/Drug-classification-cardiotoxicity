# Drug-classification-cardiotoxicity
AI-based system to detect drug-induced cardiotoxicity using multiplexed cardiac contractility features (Random Forest, XGBoost, ANN) with a Streamlit dashboard for real-time prediction.
# AI-Based Drug Classification System Using Multiplexed Cardiac Contractility Features

## Overview
Drug-induced cardiotoxicity is one of the leading causes of late-stage clinical 
trial failures and drug withdrawals. This project presents an AI-powered system 
that analyzes multiplexed cardiac contractility features — such as beat rate, 
contraction amplitude, relaxation time, and variability — to classify drugs as 
cardiotoxic or non-cardiotoxic, enabling faster and more reliable drug safety 
screening.

## Key Features
- Predicts drug-induced cardiotoxicity from cardiac contractility data
- Trained and compared multiple ML/DL models: Random Forest, XGBoost, and 
  Artificial Neural Networks (ANN)
- Flask REST API (`/predict` endpoint) for real-time predictions with 
  confidence scores
- Interactive Streamlit dashboard for dataset upload, feature visualization, 
  and prediction results
- Reduces manual effort and speeds up preclinical drug safety evaluation

## Tech Stack
Python, Flask, Streamlit, Scikit-learn, XGBoost, TensorFlow/Keras, Pandas, 
NumPy, Matplotlib

## How It Works
1. User uploads cardiac contractility dataset (CSV/Excel)
2. System preprocesses and normalizes the data
3. Key features are extracted (beat rate, amplitude, relaxation time, variability)
4. Trained ML/DL model predicts drug toxicity
5. Results are displayed with confidence score and visualizations
