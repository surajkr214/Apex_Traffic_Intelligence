# 🚦 Intelligent Traffic Management System - Apex Research

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![YOLOv8](https://img.shields.io/badge/AI-YOLOv8-green)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://apex-traffic-intelligence.streamlit.app)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)

## 🔗 Live Demo
**[Click here to use the Live Traffic Dashboard](https://apex-traffic-intelligence.streamlit.app)**

## 📋 Executive Summary
Developed for **Apex Research**, this project utilizes Computer Vision and Deep Learning to automate traffic data acquisition. By leveraging **YOLOv8** and the **VisDrone dataset**, the system detects, classifies, and counts vehicles in real-time to assist transportation authorities in congestion management and infrastructure planning.

## 🚀 Key Features
* **Custom Object Detection:** Fine-tuned YOLOv8 model detecting 10+ specific classes (Car, Truck, Bus, Pedestrian, etc.).
* **Interactive Dashboard:** A Streamlit-based web interface for video upload and real-time analytics.
* **Congestion Alerts:** Automated logic to trigger warnings when vehicle density exceeds safety thresholds.
* **Data Visualization:** Live Plotly charts displaying vehicle composition breakdown.

## 📂 Project Structure
```text
Apex_Traffic_Project/
├── app.py                  # Main Streamlit Dashboard application
├── Model_Training.ipynb    # Jupyter Notebook for Data Prep & Model Training
├── requirements.txt        # Project dependencies
├── apex_traffic_best.pt    # Trained Model Weights (Download separately)
└── README.md               # Project Documentation