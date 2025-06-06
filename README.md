# Lip-Sync Video Generation with SadTalker

This project uses [SadTalker](https://github.com/OpenTalker/SadTalker) to generate lip-synced videos from an input image and audio file via a FastAPI WebSocket server. A Streamlit client provides a user interface to upload inputs and display the output video. The system can be run locally in a Conda environment or in a Docker container.

## Table of Contents
- [Installation Instructions](#installation-instructions)
  - [Local Setup (Conda)](#local-setup-conda)
  - [Docker Setup](#docker-setup)
- [System Overview](#system-overview)
- [How to Test with a WebSocket Client](#how-to-test-with-a-websocket-client)
- [How to Run with Docker](#how-to-run-with-docker)
- [Troubleshooting](#troubleshooting)

## System Overview

The system generates lip-synced videos by processing an input image (containing a face) and an audio file (e.g., WAV) to produce a video where the face’s lip movements match the audio. The pipeline consists of:

1. **FastAPI WebSocket Server (`main.py`)**:
   - Accepts base64-encoded image and audio inputs via a WebSocket endpoint (`/ws/lipsync`).
   - Saves inputs to temporary files and runs `SadTalker/inference.py` as a subprocess.
   - Returns the generated video as base64-encoded data.

2. **Streamlit Client (`client.py`)**:
   - Provides a web interface to upload an image and audio, send them to the WebSocket server, and display the resulting video.

The system runs in the `illuminus_ai` Conda environment locally or the `lip-sync-api` environment in Docker.
## Installation Instructions

### Local Setup (Conda)

1. **Install Miniconda**:
   - Download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) for your platform (Windows/Linux/macOS).
   - On Windows, use PowerShell or Anaconda Prompt.

2. **Create Conda Environment**:
   ```powershell
   conda create -n illuminus_ai python=3.8
   conda activate illuminus_ai
   ```

3. **Install Dependencies**:
   - Create a `requirements.txt` file with the following content:
     ```text
         fastapi==0.115.0 
         uvicorn==0.30.6 
         pydantic==2.9.2 
         websockets==12.0 
         streamlit==1.38.0 
         torch==2.0.1 
         torchvision==0.15.2 
         numpy==1.24.4 
         opencv-python==4.10.0.84 
         gfpgan==1.3.8
         face_alignment==1.3.5
         imageio==2.19.3
         imageio-ffmpeg==0.4.7
         librosa==0.9.2 
         resampy==0.3.1
         pydub==0.25.1 
         scipy==1.10.1
         kornia==0.5.4
         yacs==0.1.8
         pyyaml  
         joblib==1.1.0
         scikit-image==0.19.3
         basicsr==1.4.2
         facexlib==0.3.0
         gradio
         av
         safetensors
     ```
   - Install dependencies:
     ```powershell
     pip install -r requirements.txt
     ```

4. **Install SadTalker**:
   ```powershell
   git clone https://github.com/OpenTalker/SadTalker.git
   ```

5. **Download Checkpoints**:
   - Create the `SadTalker/checkpoints` directory and download required files:
     ```powershell
     mkdir SadTalker\checkpoints
     Invoke-WebRequest -Uri https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_256.safetensors -OutFile SadTalker\checkpoints\SadTalker_V0.0.2_256.safetensors
     Invoke-WebRequest -Uri https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00109-model.pth.tar -OutFile SadTalker\checkpoints\mapping_00109-model.pth.tar
     Invoke-WebRequest -Uri https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00229-model.pth.tar -OutFile SadTalker\checkpoints\mapping_00229-model.pth.tar
     ```

6. **Apply Code Fixes**:
   - Update `SadTalker/src/face3d/util/my_awing_arch.py` to fix `np.float` deprecation (line 18):
     ```python
     preds = preds.astype(np.float32, copy=False)
     ```
   - Update `SadTalker/src/face3d/util/preprocess.py` to fix `ValueError` in `POS` (replace the `POS` function):
     ```python
     def POS(xp, x):
         npts = xp.shape[1]
         A = np.zeros([2*npts, 8])
         A[0:2*npts-1:2, 0:3] = x.transpose()
         A[0:2*npts-1:2, 3] = 1
         A[1:2*npts:2, 4:7] = x.transpose()
         A[1:2*npts:2, 7] = 1
         b = np.reshape(xp.transpose(), [2*npts, 1])
         k, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
         R1 = k[0:3]
         R2 = k[4:7]
         sTx = k[3].item()
         sTy = k[7].item()
         s = (np.linalg.norm(R1) + np.linalg.norm(R2))/2
         t = np.array([sTx, sTy])
         return t, s
     ```

7. **Verify Setup**:
   ```powershell
   python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
   ```
   Expected: `1.12.0 True` (if GPU is available).

### Docker Setup

1. **Install Docker**:
   - Download and install [Docker Desktop](https://www.docker.com/products/docker-desktop/) for Windows/Linux/macOS.
   - For GPU support, install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

2. **Create `Dockerfile`**:
   ```dockerfile
         FROM continuumio/miniconda3
         WORKDIR /app
         RUN conda create -n lip-sync-api python=3.8
         SHELL ["conda", "run", "-n", "lip-sync-api", "/bin/bash", "-c"]
         RUN apt-get update && apt-get install -y \
            ffmpeg \
            git \
            gcc \
            g++ \
            make \
            cmake \
            libc-dev \
            libpython3-dev \
            libffi-dev \
            libgl1-mesa-dev \
            && rm -rf /var/lib/apt/lists/*
         COPY requirements.txt .
         RUN conda install pytorch==2.0.1 torchvision==0.15.2 -c pytorch && \
            pip install --timeout=1000 -r requirements.txt
         RUN git clone https://github.com/OpenTalker/SadTalker.git && \
            cd SadTalker && \
            pip install -r requirements.txt
         RUN mkdir -p /app/SadTalker/checkpoints && \
            wget -P /app/SadTalker/checkpoints https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_256.safetensors && \
            wget -P /app/SadTalker/checkpoints https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00109-model.pth.tar && \
            wget -P /app/SadTalker/checkpoints https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00229-model.pth.tar
         COPY main.py .
         COPY client.py .
         COPY inference.py /app/SadTalker/inference.py
         COPY SadTalker/src/face3d/util/preprocess.py /app/SadTalker/src/face3d/util/preprocess.py
         COPY SadTalker/src/face3d/util/my_awing_arch.py /app/SadTalker/src/face3d/util/my_awing_arch.py
         EXPOSE 8000 8501
         CMD ["bash", "-c", "conda run -n lip-sync-api uvicorn main:app --host 0.0.0.0 --port 8000 --ws-ping-timeout 120 --ws-ping-interval 60 & conda run -n lip-sync-api streamlit run client.py --server.port 8501 --server.address 0.0.0.0"]
   ```

3. **Build and Run Docker Container**:
   ```powershell
   docker build -t lipsync-api .
   docker run -p 8000:8000 -p 8501:8501 lipsync-api   
   ```

4. **Verify Docker Setup**:
   - Access the Streamlit client at `http://localhost:8501`.
   - Check container logs:
     ```powershell
     docker logs <container-id>
     ```

## How to Test with a WebSocket Client

1. **Prepare a Test Script (`main.py`)**:

2. **Prepare Test Files**:
   - Place a test image (`sample_image.png`) and audio (`sample_audio.wav`) in `default folder`.

3. **Run the Server**:
   ```powershell
   conda activate illuminus_ai
   python main.py
   ```

4. **Run the client Test Script**:
   ```powershell
   streamlit run client.py
   ```
   - Browse the test image and audio.
   - Wait for the server to process the input and send the response back to the client.
   - Video file will appear on test client.
   - Output: `.mp4`  will be saved in `temp/output` if successful.

## How to Run with Docker

1. **Build the Docker Image**:
   ```powershell
   docker build -t lipsync-api .
   ```

2. **Run the Container**:
   - With GPU support:
     ```powershell
     docker run --gpus all -p 8000:8000 -p 8501:8501 lipsync-api
     ```
   - Without GPU:
     ```powershell
     docker run -p 8000:8000 -p 8501:8501 lipsync-api
     ```

3. **Access the Application**:
   - **FastAPI WebSocket**: Connect to `ws://localhost:8000/ws/lipsync`.
   - **Streamlit Client**: Open `http://localhost:8501` in a browser to upload image/audio and view results.

4. **Test with WebSocket Client**:
   - Follow the [WebSocket client instructions](#how-to-test-with-a-websocket-client) while the container is running.

## Troubleshooting

- **ValueError in `preprocess.py`**:
  - Verify `test.png` contains a clear face.
  - Ensure `preprocess.py` has the updated `POS` function.

- **Docker Issues**:
  - Check logs: `docker logs <container-id>`.
  - Ensure NVIDIA drivers and Container Toolkit are installed for GPU support.

For further assistance, provide:
- Error logs from `main.py` or `inference.py`.
- Hardware specs (CPU, GPU, RAM).