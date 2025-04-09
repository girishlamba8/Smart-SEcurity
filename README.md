#  Surveillance Dashboard

An intelligent video surveillance system that integrates **Crowd Detection**, **Work Monitoring**, and **Crime Prevention** using YOLOv5 and DeepSORT. The system can process uploaded video files or live CCTV stream links, offering real-time analytics through a user-friendly web dashboard built with Streamlit.

---

##  Features

###  Combined Web UI
- Developed with Streamlit.
- Allows selecting one or more modules to process videos or streams.
- Supports both file upload and CCTV stream URLs.

###  Crowd Detection
- Uses YOLOv5 for detecting people.
- DeepSORT for multi-object tracking.
- Displays real-time person count and tracking IDs.

###  Work Monitoring
- Define custom polygonal work zones using mouse input.
- Detects if any zone becomes empty and raises alerts.
- Zones are persisted using JSON, so they reload on future runs.

###  Crime Prevention (Fight Detection)
- Uses proximity-based logic to detect potential fights.
- Visual and text alerts on frame when suspicious activity is detected.

---

## 🗂 Project Structure

```bash

├── combined_module.py       # Streamlit-based UI dashboard
├── crowd_detection.py       # Crowd counting and tracking
├── work_monitoring.py       # Work zone monitoring system
├── crime_prevention.py      # Simple fight detection system
└── saved_polygons.json      # (Auto-generated) Stores defined zones

