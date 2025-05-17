ARG REG=docker.io
FROM ${REG}/dayuhub/rtsp-server

LABEL authors="skyrim"

ARG dependency_dir=dependency
ARG lib_dir=dependency/core/lib
ARG code_dir=datasource

ENV TZ=Asia/Shanghai

RUN apt-get update && apt-get install -y \
    wget \
    tar \
    xz-utils \
    libxext6 \
    libsm6 \
    libxrender1 \
    libgl1 \
    ffmpeg \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-tools \
    gstreamer1.0-alsa \
    gstreamer1.0-pulseaudio \
    net-tools \
    netcat \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY ${lib_dir}/requirements.txt ./lib_requirements.txt
COPY ${code_dir}/requirements.txt ./code_requirements.txt

RUN apt-get remove -y --ignore-missing python3-yaml || true && \
    pip3 install --upgrade pip && \
    pip3 install -r lib_requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip3 install -r code_requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY  ${dependency_dir} /home/dependency
ENV PYTHONPATH="/home/dependency"

COPY  ${code_dir}/* /app/

CMD ["python3", "main.py"]
