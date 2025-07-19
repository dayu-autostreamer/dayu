ARG REG=docker.io
FROM ${REG}/ultralytics/ultralytics:latest

LABEL authors="Wenhui Zhou"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

RUN sed -i 's@http://archive.ubuntu.com/ubuntu/@https://mirrors.tuna.tsinghua.edu.cn/ubuntu/@g' /etc/apt/sources.list \
 && sed -i 's@http://security.ubuntu.com/ubuntu/@https://mirrors.tuna.tsinghua.edu.cn/ubuntu/@g' /etc/apt/sources.list

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      wget gnupg ca-certificates \
      nvidia-cuda-toolkit \
      libnvinfer8 libnvinfer-dev libnvinfer-plugin8 python3-libnvinfer \
      libnvonnxparsers8 libnvparsers8 \
      python3-pycuda \
 && apt-mark hold libnvinfer8 libnvinfer-dev libnvinfer-plugin8 libnvonnxparsers8 libnvparsers8 \
 && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --upgrade pip \
 && pip3 install --no-cache-dir \
      typing_extensions scipy tiff imagecodecs \
      scikit-learn scikit-image ptflops numpy==1.23.1

CMD ["/bin/bash"]
