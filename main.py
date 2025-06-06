import asyncio
import base64
import json
import os
from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from fastapi import WebSocketDisconnect


app = FastAPI()
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)


class LipSyncInput(BaseModel):
    image: str
    audio: str


async def run_sadtalker(image_path: str, audio_path: str, output_path: str):
    try:
        sadtalker_cmd = [
            "python", "SadTalker/inference.py",
            "--driven_audio", audio_path,
            "--source_image", image_path,
            "--result_dir", output_path,
            "--still",
            "--preprocess", "crop",
            "--enhancer", "gfpgan",
            "--checkpoint_dir", "SadTalker/checkpoints",
        ]
        process = await asyncio.create_subprocess_exec(
            *sadtalker_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            raise Exception(f"SadTalker failed: {stderr.decode()}")

        for file in os.listdir(output_path):
            if file.endswith(".mp4"):
                return os.path.join(output_path, file)
        raise Exception("No video file generated")
    except Exception as e:
        raise Exception(f"SadTalker processing error: {str(e)}")


@app.websocket("/ws/lipsync")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            try:
                data = await websocket.receive_text()
                input_data = json.loads(data)
                input_model = LipSyncInput(**input_data)

                # Decode inputs
                image_data = base64.b64decode(input_model.image)
                audio_data = base64.b64decode(input_model.audio)

                # Save inputs
                image_path = os.path.join(
                    TEMP_DIR, f"input_{uuid.uuid4()}.png")
                audio_path = os.path.join(
                    TEMP_DIR, f"input_{uuid.uuid4()}.wav")
                output_dir = os.path.join(TEMP_DIR, "output")

                with open(image_path, "wb") as f:
                    f.write(image_data)
                with open(audio_path, "wb") as f:
                    f.write(audio_data)

                os.makedirs(output_dir, exist_ok=True)

                # Run SadTalker
                video_path = await run_sadtalker(image_path, audio_path, output_dir)

                # Encode video to base64
                with open(video_path, "rb") as f:
                    video_data = base64.b64encode(f.read()).decode()

                # Send response
                response = {
                    "video": video_data,
                    "status": "success",
                    "message": ""
                }
                await websocket.send_json(response)

            except Exception as e:
                await websocket.send_json({
                    "video": "",
                    "status": "error",
                    "message": str(e)
                })
    except WebSocketDisconnect as e:
        print(f"WebSocket disconnected: {e.code} {e.reason}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
    finally:
        try:
            await websocket.close()
        except RuntimeError:
            pass


@app.get("/health")
async def health_check():
    return JSONResponse({"status": "healthy"})


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000,
                ws_ping_timeout=120, ws_ping_interval=60)
