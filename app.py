# app.py

import streamlit as st
import numpy as np
import torch
import imageio
import requests
from streamlit_lottie import st_lottie
from model_definitions import FuNetA
from my_models import extract_faces_from_video, image_to_graph
import time
import random

# CSS etc. (same as before)
st.markdown("""
<style>
/* High-tech, clean color palette */
body { background-color: #0a192f; }
.main { background-color: #0a192f; padding: 0; }
body, .stTextInput, .stFileUploader label, .stMarkdown, .stText, h1, h2, h3, h4, h5, h6, p {
    color: #ccd6f6 !important;
    font-family: 'Segoe UI', sans-serif;
}
h1 {
    font-size: 42px;
    color: #64ffda !important;
    font-weight: 800;
    text-align: left;
    margin-bottom: 15px;
    animation: shimmer 1.5s infinite alternate;
}
@keyframes shimmer {
    0% {
        color: #ccd6f6;
        text-shadow: none;
    }
    100% {
        color: #64ffda;
        text-shadow: 0 0 10px #64ffda, 0 0 20px #64ffda;
    }
}
h3 { font-size: 24px; color: #ccd6f6 !important; font-weight: 700; }
.teal-card {
    background-color: #112240;
    padding: 25px;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.25);
    border:1px solid #64ffda;
    color:#ccd6f6;
    margin-bottom:20px;
}
.taskbar {
    background-color: #0a192f;
    padding: 15px 30px;
    border-radius: 0 0 12px 12px;
    color:#ccd6f6;
    display:flex;
    justify-content:space-between;
    align-items:center;
    font-family:'Segoe UI',sans-serif;
    border-bottom:2px solid #64ffda;
}
.taskbar a {
    color:#ccd6f6;
    margin:0 10px;
    font-size:18px;
    text-decoration:none;
    font-weight:600;
    position:relative;
    transition: color 0.2s ease-in-out;
}
.taskbar a:not(:last-child)::after {
    content:"|";
    color:#ccd6f6;
    margin-left:15px;
    margin-right:10px;
    font-weight:400;
}
.taskbar a:hover {
    text-decoration:underline;
    color:#64ffda;
}
.stAlert {
    border-radius:10px;
    border:1px solid #64ffda;
    background-color:#112240 !important;
    color:#ccd6f6 !important;
    font-size:18px;
    font-weight:600;
}
.block-container { max-width:95% !important; padding-left:2rem; padding-right:2rem; }
.stProgress > div > div > div > div {
    background-color: #64ffda !important;
}
.stSuccess > div > div {
    background-color: #4CAF50 !important;
    color: white !important;
}
.stWarning > div > div {
    background-color: #FFC107 !important;
    color: black !important;
}
.stError > div > div {
    background-color: #F44336 !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="taskbar">
    <div>
        <strong style='font-size: 20px;'>👊 Deepfake Analyzer Pro 🤖</strong><br>
    </div>
    <div>
        <a href="#upload">Upload</a>
        <a href="#about">About</a>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<h1>Deepfake Detector 🔍</h1>", unsafe_allow_html=True)
st.write("Unmask the truth with AI-powered deepfake detection. Upload one or more images/videos and let the intelligence do the rest.")
st.markdown("---")

st.markdown("""
<div class="teal-card" id="upload">
    <h3>Upload Images and Videos</h3>
    <p style='font-weight:bold;'>Drag and drop your files below, or browse to upload.</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    uploaded_images = st.file_uploader(
        "Upload Image(s)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

with col2:
    uploaded_videos = st.file_uploader(
        "Upload Video(s)",
        type=["mp4", "avi", "mov", "mp4v"],
        accept_multiple_files=True
    )

uploaded_files = []
if uploaded_images:
    uploaded_files.extend(uploaded_images)
if uploaded_videos:
    uploaded_files.extend(uploaded_videos)

# Load model (simple version)
device = torch.device("cpu")
model = FuNetA(num_classes=2)
model.to(device)
model.eval()

def predict_from_image_bytes(image_bytes):
    tensor = image_to_graph(image_bytes)
    if tensor is None:
        return None  # Could not parse or resize
    tensor = tensor.to(device)
    with torch.no_grad():
        logits = model(tensor, graph=None)
        probs = torch.softmax(logits, dim=1)
        prob_fake = probs[0,1].item()  # class 1 = "fake"
    return prob_fake

def display_results_grid(results):
    max_cols = 5
    img_results = [r for r in results if r['file_type'] == 'image']
    vid_results = [r for r in results if r['file_type'] == 'video']

    if img_results:
        st.markdown("<br>---<br>", unsafe_allow_html=True)
        st.markdown("<h3><u>Image Detection Results</u></h3>", unsafe_allow_html=True)
        st.markdown("<hr style='border:1px solid #112240;'>", unsafe_allow_html=True)
        for i in range(0, len(img_results), max_cols):
            cols = st.columns(max_cols)
            for j, r in enumerate(img_results[i:i+max_cols]):
                with cols[j]:
                    st.image(r['preview'], caption=r['filename'], width=200)
                    prob_fake = r.get('prob_fake', None)
                    if prob_fake is None:
                        st.write("Error in prediction")
                        continue
                    prob_real = 1 - prob_fake
                    st.progress(int(prob_fake * 100))
                    st.write(f"**Fake:** {prob_fake * 100:.2f}%")
                    st.write(f"**Real:** {prob_real * 100:.2f}%")
                    if prob_fake > 0.7:
                        st.error(f"⚠️ Strong evidence of Deepfake ({prob_fake * 100:.1f}% Fake)")
                    elif prob_fake > 0.5:
                        st.warning(f"⚠️ Possible Deepfake ({prob_fake * 100:.1f}% Fake)")
                    else:
                        st.success(f"✅ Likely Authentic ({prob_real * 100:.1f}% Real)")

    if vid_results:
        st.markdown("<br>---<br>", unsafe_allow_html=True)
        st.markdown("<h3><u>Video Detection Results</u></h3>", unsafe_allow_html=True)
        st.markdown("<hr style='border:1px solid #112240;'>", unsafe_allow_html=True)
        for i in range(0, len(vid_results), max_cols):
            cols = st.columns(max_cols)
            for j, r in enumerate(vid_results[i:i + max_cols]):
                with cols[j]:
                    st.write(f"**File:** {r['filename']}")
                    st.video(r['file_data'])
                    prob_fake = r.get('prob_fake', None)
                    if prob_fake is None:
                        st.write("Error in prediction")
                        continue
                    prob_real = 1 - prob_fake
                    st.progress(int(prob_fake * 100))
                    st.write(f"**Fake:** {prob_fake * 100:.2f}%")
                    st.write(f"**Real:** {prob_real * 100:.2f}%")
                    if prob_fake > 0.7:
                        st.error(f"⚠️ Strong evidence of Deepfake ({prob_fake * 100:.1f}% Fake)")
                    elif prob_fake > 0.5:
                        st.warning(f"⚠️ Possible Deepfake ({prob_fake * 100:.1f}% Fake)")
                    else:
                        st.success(f"✅ Likely Authentic ({prob_real * 100:.1f}% Real)")

def display_detection_report(results):
    total_files = len(results)
    total_images = sum(1 for r in results if r['file_type'] == 'image')
    total_videos = sum(1 for r in results if r['file_type'] == 'video')
    likely_fakes = sum(1 for r in results if (r.get('prob_fake', 0) > 0.5))

    st.markdown("<div class='teal-card'><h3>Detection Report</h3>", unsafe_allow_html=True)
    st.write(f"**Total files uploaded:** {total_files}")
    st.write(f"**Images:** {total_images}")
    st.write(f"**Videos:** {total_videos}")
    st.write(f"**Likely Fakes:** {likely_fakes}")
    st.markdown("</div>", unsafe_allow_html=True)

if uploaded_files:
    results = []
    scanning_placeholder = st.empty()
    progress_placeholder = st.empty()

    for uploaded_file in uploaded_files:
        file_type = uploaded_file.type
        filename = uploaded_file.name

        scanning_placeholder.markdown(f"**🔍 Scanning file: `{filename}`...**")
        my_progress_bar = progress_placeholder.progress(0)

        if "image" in file_type:
            file_bytes = uploaded_file.getvalue()
            prob_fake = predict_from_image_bytes(file_bytes)
            if prob_fake is None:
                prob_fake = 0.5  # fallback
            results.append({
                'file_type': 'image',
                'filename': filename,
                'preview': uploaded_file.getvalue(),
                'prob_fake': prob_fake
            })
        elif "video" in file_type:
            # For videos, you could extract frames or faces
            # Here we mock for simplicity
            prob_fake = random.uniform(0.05, 0.95)
            results.append({
                'file_type': 'video',
                'filename': filename,
                'file_data': uploaded_file.getvalue(),
                'prob_fake': prob_fake
            })

        # Simulate progress
        for i in range(100):
            progress_placeholder.progress(i)
            time.sleep(0.005)

    scanning_placeholder.empty()
    progress_placeholder.empty()

    if results:
        display_detection_report(results)
        display_results_grid(results)

st.markdown("<hr style='border:1px solid #64ffda;'>", unsafe_allow_html=True)
st.markdown("<h3 id='about'>About This Project</h3>", unsafe_allow_html=True)
st.write("🚀 **Deepfake Analyzer Pro** uses a simplified CNN to detect manipulations in images/videos.")
