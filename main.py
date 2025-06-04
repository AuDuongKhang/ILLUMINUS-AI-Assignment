from fastapi import FastAPI, WebSocket
import base64
import os
import cv2
import librosa
import uvicorn
from Wav2Lip.inference import run_inference as wav2lip_inference

app = FastAPI()

TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)


@app.websocket("/lip-sync")
async def lip_sync_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        audio_b64 = data.get("audio")
        image_b64 = data.get("image")

        # Decode base64 inputs
        audio_data = base64.b64decode(audio_b64)
        image_data = base64.b64decode(image_b64)

        # Save inputs to temporary files
        audio_path = os.path.join(TEMP_DIR, "input_audio.wav")
        image_path = os.path.join(TEMP_DIR, "input_image.png")
        with open(audio_path, "wb") as f:
            f.write(audio_data)
        with open(image_path, "wb") as f:
            f.write(image_data)

        # Preprocess inputs (e.g., ensure image has a face, normalize audio)
        img = cv2.imread(image_path)
        if img is None:
            await websocket.send_json({"error": "Invalid image"})
            return
        # Wav2Lip expects 16kHz audio
        audio, sr = librosa.load(audio_path, sr=16000)
        if len(audio) == 0:
            await websocket.send_json({"error": "Invalid audio"})
            return

        # Run Wav2Lip inference
        output_video_path = os.path.join(TEMP_DIR, "output_video.mp4")
        wav2lip_inference(
            checkpoint_path="models/Wav2Lip-SD-GAN.pt",
            face=image_path,
            audio=audio_path,
            outfile=output_video_path
        )

        with open(output_video_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode("utf-8")

        # Send response
        await websocket.send_json({"video": video_b64})

        os.remove(audio_path)
        os.remove(image_path)
        os.remove(output_video_path)

    except Exception as e:
        await websocket.send_json({"error": str(e)})
    finally:
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
