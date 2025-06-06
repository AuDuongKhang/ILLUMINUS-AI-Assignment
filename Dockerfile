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