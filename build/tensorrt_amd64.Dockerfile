ARG REG=docker.io
FROM ${REG}/ultralytics/ultralytics:latest

LABEL authors="Wenhui Zhou"

ENV DEBIAN_FRONTEND=noninteractive

RUN sed -i 's@http://archive.ubuntu.com/ubuntu/@https://mirrors.tuna.tsinghua.edu.cn/ubuntu/@g' /etc/apt/sources.list && \
    sed -i 's@http://security.ubuntu.com/ubuntu/@https://mirrors.tuna.tsinghua.edu.cn/ubuntu/@g' /etc/apt/sources.list

RUN apt-get update && apt-get install -y --no-install-recommends wget gnupg ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN wget -qO /tmp/cuda-keyring.deb \
      https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i /tmp/cuda-keyring.deb && \
    rm -f /tmp/cuda-keyring.deb && \
    rm -f /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update && apt-get install -y --no-install-recommends \
      nvidia-cuda-toolkit \
      libnvinfer8 libnvinfer-dev libnvinfer-plugin8 python3-libnvinfer \
      libnvonnxparsers8 libnvparsers8 && \
    apt-mark hold libnvinfer8 libnvinfer-dev libnvinfer-plugin8 libnvonnxparsers8 libnvparsers8 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip && \
    pip3 install typing_extensions scipy tiff imagecodecs \
      scikit-learn scikit-image pycuda numpy==1.23.1 \
      tensorrt -i https://pypi.tuna.tsinghua.edu.cn/simple
