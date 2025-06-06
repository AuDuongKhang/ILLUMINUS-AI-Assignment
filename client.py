import streamlit as st
import base64
import json
import websockets
import asyncio

st.title("Lip-Sync Video Generator")

image_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
audio_file = st.file_uploader("Upload Audio", type=["wav"])


async def send_to_websocket(image_data, audio_data):
    uri = "ws://localhost:8000/ws/lipsync"
    try:
        async with websockets.connect(uri, ping_interval=60, ping_timeout=120) as websocket:
            image_b64 = base64.b64encode(image_data).decode()
            audio_b64 = base64.b64encode(audio_data).decode()
            payload = {"image": image_b64, "audio": audio_b64}
            await websocket.send(json.dumps(payload))
            response = await websocket.recv()
            return json.loads(response)
    except websockets.exceptions.ConnectionClosed as e:
        st.error(f"WebSocket connection closed: {str(e)}")
        return {"status": "error", "message": str(e)}

if st.button("Generate Video"):
    if image_file and audio_file:
        with st.spinner("Generating lip-synced video..."):
            try:
                image_data = image_file.read()
                audio_data = audio_file.read()
                response = asyncio.run(
                    send_to_websocket(image_data, audio_data))
                if response["status"] == "success":
                    video_data = base64.b64decode(response["video"])
                    st.video(video_data)
                else:
                    st.error(f"Error: {response['message']}")
            except Exception as e:
                st.error(f"Client error: {str(e)}")
    else:
        st.warning("Please upload both an image and an audio file.")
