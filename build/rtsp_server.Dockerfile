ARG REG=docker.io
FROM ${REG}/python:3.8

LABEL authors="skyrim"

ENV TZ=Asia/Shanghai

RUN apt-get update && \
    apt-get install -y ffmpeg && \
    apt-get clean

WORKDIR /rtsp_server

ENV RTSP_PATH="/rtsp_server"

ARG TARGETPLATFORM
RUN ARCH=$(echo ${TARGETPLATFORM} | cut -d'/' -f2) && \
    case "$ARCH" in \
        amd64) ARCH_TYPE="linux_amd64";; \
        arm64) ARCH_TYPE="linux_arm64v8";; \
        *) echo "Unsupported architecture: $ARCH" && exit 1;; \
    esac && \
    wget -O mediamtx.tar.gz https://github.com/bluenviron/mediamtx/releases/download/v1.9.3/mediamtx_v1.9.3_${ARCH_TYPE}.tar.gz && \
    tar -zxvf mediamtx.tar.gz

RUN sed -i \
  -e 's/rtspAddress: :8554/rtspAddress: :8000/' \
  -e 's/rtpAddress: :8000/rtpAddress: :7000/' \
  -e 's/rtcpAddress: :8001/rtcpAddress: :7001/'  \
  -e 's/writeQueueSize: 512/writeQueueSize: 32768/' \
  mediamtx.yml


CMD ["/bin/bash"]
