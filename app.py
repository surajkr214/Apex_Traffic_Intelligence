import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO
from collections import defaultdict
import pandas as pd
import plotly.express as px

# ===========================
# 1. PAGE CONFIGURATION
# ===========================
st.set_page_config(
    page_title="Apex Intelligent Traffic Dashboard",
    page_icon="🚦",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
    <style>
        .main { background-color: #0e1117; }
        .stMetric { background-color: #262730; padding: 10px; border-radius: 5px; }
        h1, h2, h3 { color: #ffffff; }
    </style>
""", unsafe_allow_html=True)

# ===========================
# 2. SIDEBAR & SETTINGS
# ===========================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1085/1085961.png", width=80)
    st.title("Apex Traffic AI")
    st.write("---")
    
    st.subheader("Settings")
    # Confidence slider
    conf_thres = st.slider("Confidence Threshold", 0.1, 1.0, 0.35, 0.05)
    # Congestion threshold input
    congestion_limit = st.number_input("Congestion Threshold (Car Count)", min_value=5, value=15)
    
    st.write("---")
    st.info("Upload a video to start.")

# ===========================
# 3. MAIN DASHBOARD UI
# ===========================
st.title("🚦 Intelligent Traffic Management System")
st.markdown("### Real-time Vehicle Detection & Congestion Monitoring")

# Load Model
@st.cache_resource
def load_model():
    # Make sure 'apex_traffic_best.pt' is in the SAME folder as app.py
    model_path = "apex_traffic_best.pt"
    
    if os.path.exists(model_path):
        return YOLO(model_path)
    else:
        st.warning("⚠️ 'apex_traffic_best.pt' not found. Using standard YOLOv8n for demo.")
        return YOLO("yolov8n.pt")

model = load_model()

# File Uploader
uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save uploaded file to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    
    # Layout: Video on Left, Stats on Right
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st_frame = st.empty()
        
    with col2:
        st.subheader("Live Statistics")
        kpi1, kpi2 = st.columns(2)
        with kpi1:
            st_car = st.empty()
        with kpi2:
            st_total = st.empty()
        
        st.write("---")
        st.subheader("Vehicle Breakdown")
        st_chart = st.empty()
        
        st.write("---")
        st_alert = st.empty()

    vehicle_counts = defaultdict(int)
    frame_count = 0  # FIX 1: Initialize frame counter
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        frame_count += 1
        
        # Resize frame for performance (keeps app running smoothly)
        frame = cv2.resize(frame, (720, 480))
        
        # Inference
        results = model.predict(frame, conf=conf_thres, verbose=False)
        result = results[0]
        
        # Count vehicles
        current_counts = defaultdict(int)
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = result.names[cls_id]
            current_counts[label] += 1
        
        car_count = current_counts.get('car', 0)
        total_vehicles = sum(current_counts.values())

        # Update Metrics
        with st_car:
            st.metric(label="Active Cars", value=car_count)
        with st_total:
            st.metric(label="Total Vehicles", value=total_vehicles)
            
        # Update Chart (Real-time Bar Chart)
        df_chart = pd.DataFrame({
            "Vehicle": list(current_counts.keys()),
            "Count": list(current_counts.values())
        })
        
        if not df_chart.empty:
            fig = px.bar(df_chart, x="Vehicle", y="Count", color="Vehicle", height=250)
            
            # FIX 2: Use frame_count in the key to make it unique every time
            st_chart.plotly_chart(fig, use_container_width=True, key=f"chart_{frame_count}")

        # Draw Bounding Boxes
        annotated_frame = result.plot()
        
        # CONGESTION ALERT LOGIC
        if car_count > congestion_limit:
            st_alert.error(f"🚨 CONGESTION ALERT! Car count ({car_count}) exceeds limit ({congestion_limit})")
            # Red overlay on video frame
            cv2.rectangle(annotated_frame, (0, 0), (720, 50), (0, 0, 255), -1)
            cv2.putText(annotated_frame, "CONGESTION ALERT", (250, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            st_alert.success("✅ Traffic Flow Normal")

        # Convert to RGB for Streamlit display
        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        
        # FIX 3: Replaced 'use_column_width' with 'use_container_width'
        st_frame.image(frame_rgb, channels="RGB", use_container_width=True)

    cap.release()