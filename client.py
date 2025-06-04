import streamlit as st
import websockets
import json
import base64
import asyncio
import os
import cv2
import librosa
import numpy as np


# Connect to WebSocket server
async def send_to_server(audio_b64, image_b64):
    uri = "ws://localhost:8000/lip-sync"
    async with websockets.connect(uri, ping_interval=20, ping_timeout=20) as websocket:
        await websocket.send(json.dumps({"audio": audio_b64, "image": image_b64}))
        response = await websocket.recv()
    return json.loads(response)


def main():
    st.set_page_config(page_title="Lip-Sync WebSocket Client")

    st.title("Lip-Sync WebSocket Client")
    st.write(
        "Upload an audio (WAV, 16kHz) and image (PNG/JPG) to generate a lip-synced video.")

    # File upload
    audio_file = st.file_uploader(
        "Upload Audio File (WAV, 16kHz)", type=["wav"])
    image_file = st.file_uploader(
        "Upload Image File (PNG/JPG)", type=["png", "jpg", "jpeg"])

    if st.button("Generate Lip-Synced Video"):
        if audio_file is None or image_file is None:
            st.error("Please upload both an audio and an image file.")
        else:
            with st.spinner("Processing..."):
                try:
                    # Save uploaded files temporarily
                    audio_path = "temp_audio.wav"
                    image_path = "temp_image.png"
                    with open(audio_path, "wb") as f:
                        f.write(audio_file.read())
                    with open(image_path, "wb") as f:
                        f.write(image_file.read())

                    # Validate audio
                    try:
                        audio, sr = librosa.load(audio_path, sr=None)
                        if sr != 16000:
                            st.warning(
                                f"Audio sample rate is {sr}Hz, Wav2Lip expects 16kHz. Results may vary.")
                    except Exception as e:
                        st.error(f"Invalid audio file: {str(e)}")
                        return

                    # Validate image
                    img = cv2.imread(image_path)
                    if img is None:
                        st.error(f"Invalid image file: {image_path}")
                        return

                    face_cascade = cv2.CascadeClassifier(
                        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                    if len(faces) == 0:
                        st.error("No face detected in the image.")
                        return
                    if len(faces) > 1:
                        st.warning(
                            "Multiple faces detected, Wav2Lip expects a single face. Results may vary.")

                    # Encode files to base64
                    with open(audio_path, "rb") as f:
                        audio_b64 = base64.b64encode(f.read()).decode("utf-8")
                    with open(image_path, "rb") as f:
                        image_b64 = base64.b64encode(f.read()).decode("utf-8")

                    # Run async WebSocket call
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    response = loop.run_until_complete(
                        send_to_server(audio_b64, image_b64))
                    loop.close()

                    # Handle response
                    if "video" in response:
                        output_path = "output_video.mp4"
                        with open(output_path, "wb") as f:
                            f.write(base64.b64decode(response["video"]))
                        st.success(
                            f"Video generated successfully! Saved as {output_path}")
                        st.video(output_path)
                    else:
                        st.error(f"Server error: {response.get('error')}")

                    os.remove(audio_path)
                    os.remove(image_path)

                except Exception as e:
                    st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
