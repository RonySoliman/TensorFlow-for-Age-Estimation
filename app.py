# app.py - Enhanced Version
import streamlit as st
import cv2
import tempfile
import os
import shutil
import subprocess
import numpy as np
from image_processing import process_frame
from feature_extraction import get_performance_metrics, detect_mask
from datetime import datetime

# ... existing imports ...

# Initialize session state
if 'processing_done' not in st.session_state:
    st.session_state.processing_done = False
    st.session_state.metrics = None
    st.session_state.start_time = None
    st.session_state.output_video_path = None
    st.session_state.output_video_mp4 = None

# REMOVED: st.session_state.annotated_frames - too memory heavy

# ... custom CSS remains ...

# ===== SIDEBAR =====
# ... unchanged sidebar code ...

# ===== PROCESSING FUNCTIONS =====
def apply_augmentation(frame):
    """Lightweight augmentation"""
    if not data_augmentation:
        return frame

    try:
        # Only apply fast augmentations
        if np.random.rand() > 0.5:
            frame = cv2.flip(frame, 1)
        
        # Lightweight brightness adjustment
        alpha = np.random.uniform(0.9, 1.1)
        frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=0)
        
    except Exception:
        pass
        
    return frame

# REMOVED: display_live_metrics() - too resource heavy

# ===== VIDEO PROCESSING =====
if uploaded_file and st.button("🚀 Process Video", type="primary"):
    # Initialize processing
    st.session_state.start_time = datetime.now()
    st.session_state.processing_done = False
    st.session_state.metrics = {"accuracy": 0, "precision": 0, "recall": 0}
    st.session_state.output_video_path = None
    st.session_state.output_video_mp4 = None
    
    # Create output directories
    for d in ["output", "output/masks"]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)
    
    # Save video to temp file
    with st.spinner("Initializing..."):
        video_bytes = uploaded_file.read()
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(video_bytes)
        video_path = tfile.name
    
    # Process video
    with st.spinner("Analyzing video..."):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Get video dimensions (downscale by 50%)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * 0.5)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 0.5)
        dim = (int(width), int(height))
        
        # Create video writer
        output_video_path = os.path.join("output", "annotated_video.avi")
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(output_video_path, fourcc, fps/2, dim)  # Half FPS
        
        if not out.isOpened():
            st.error("Failed to initialize video writer!")
            st.stop()
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        known_outfits = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Skip frames according to sampling rate
            if frame_count % frame_skip != 0:
                frame_count += 1
                continue
                
            # Downscale frame
            small_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
                
            # Apply augmentation
            small_frame = apply_augmentation(small_frame)
            
            # Process frame
            annotated_frame, _ = process_frame(
                small_frame, 
                known_outfits,
                frame_index=frame_count,
                outfit_threshold=outfit_threshold
            )
            
            # Write to video
            out.write(annotated_frame)
            
            # Update progress
            progress = frame_count / (total_frames / frame_skip)
            progress_bar.progress(min(progress, 1.0))
            status_text.text(f"Processed {frame_count} frames")
            frame_count += 1
            
            # Break early if processing too long
            if (datetime.now() - st.session_state.start_time).seconds > 300:  # 5 min limit
                st.warning("Stopped early due to time limits")
                break
        
        cap.release()
        out.release()
        progress_bar.empty()
        status_text.empty()
        
        # Finalize processing
        st.session_state.processing_done = True
        st.session_state.output_video_path = output_video_path
        processing_time = (datetime.now() - st.session_state.start_time).total_seconds()
        
        st.success(f"Processing Complete! Time: {processing_time:.1f}s")

# ===== RESULTS DISPLAY =====
if st.session_state.processing_done:
    st.header("Results")
    
    # Only show video tab
    tab1 = st.tabs(["Annotated Video"])[0]
    
    with tab1:
        video_path = st.session_state.output_video_path
        
        try:
            with open(video_path, 'rb') as video_file:
                video_bytes = video_file.read()
            
            st.video(video_bytes)
            
            st.download_button(
                label="Download Video",
                data=video_bytes,
                file_name="annotated_video.avi",
                mime="video/x-msvideo"
            )
        except Exception as e:
            st.error(f"Video error: {str(e)}")
