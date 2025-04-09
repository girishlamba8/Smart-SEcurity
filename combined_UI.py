import streamlit as st
import os
import subprocess
import tempfile

st.set_page_config(page_title="Surveillance Dashboard", layout="centered")

st.title("🛡️ Surveillance Dashboard")
st.markdown("Select the modules to run on uploaded video or CCTV stream.")

# --- Module Checkboxes ---
crowd = st.checkbox("👥 Crowd Detection")
work = st.checkbox("👷 Work Monitoring")
crime = st.checkbox("🚨 Crime Prevention")

# --- Input Section ---
source_type = st.radio("Select Input Source:", ("Upload a Video", "Paste CCTV Stream Link"))

video_path = None
temp_video_path = None

if source_type == "Upload a Video":
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
    if uploaded_file:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_file.write(uploaded_file.read())
        temp_video_path = temp_file.name
        st.success("Video uploaded successfully.")
elif source_type == "Paste CCTV Stream Link":
    video_path = st.text_input("Enter CCTV stream link (rtsp/http):")

# --- Start Button ---
if st.button("▶️ Run Selected Modules"):
    if (video_path or temp_video_path) and (crowd or work or crime):
        run_path = video_path if video_path else temp_video_path
        st.info(f"Starting modules on: `{run_path}`")

        if crowd:
            subprocess.Popen(["python", "crowd_detection.py", run_path])
        if work:
            subprocess.Popen(["python", "work_monitoring.py", run_path])
        if crime:
            subprocess.Popen(["python", "crime_prevention.py", run_path])
    else:
        st.error("Please upload a video or provide a link and select at least one module.")
