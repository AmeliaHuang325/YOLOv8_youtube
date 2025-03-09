import asyncio

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.run(asyncio.sleep(0))

import streamlit as st
import cv2
import numpy as np
import time
import yt_dlp
from ultralytics import YOLO

st.title("YOLO Object Detection on YouTube Videos")

# Load YOLO model
model = YOLO("yolov8n.pt")

# Input for YouTube video
youtube_url = st.text_input("Enter YouTube video URL:")

def extract_youtube_video_id(url):
    """Extract the video ID from a YouTube URL."""
    if "youtube.com" in url:
        return url.split("v=")[-1].split("&")[0]
    elif "youtu.be" in url:
        return url.split("/")[-1]
    return None

if youtube_url:
    video_id = extract_youtube_video_id(youtube_url)

    if video_id:
        st.markdown(f"""
            <iframe width="700" height="400" src="https://www.youtube.com/embed/{video_id}" 
            frameborder="0" allowfullscreen></iframe>
        """, unsafe_allow_html=True)

        st.success("Video embedded successfully! 🎥 Now processing frames...")

        try:
            ydl_opts = {
                "quiet": True,
                "noplaylist": True,
                "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]",
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                video_info = ydl.extract_info(youtube_url, download=False)
                video_url = video_info["url"]

            cap = cv2.VideoCapture(video_url)

            frame_window = st.empty()  # Placeholder for displaying frames

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = model(frame)[0]

                for box in results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())
                    label = results.names[cls]

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_window.image(frame_rgb, channels="RGB")

                time.sleep(1)  # Process every 1 second to reduce load

            cap.release()
        except Exception as e:
            st.error(f"Error processing video: {e}")

    else:
        st.error("Invalid YouTube URL. Please enter a valid video link.")
