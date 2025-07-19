ARG REG=docker.io
FROM ${REG}/ultralytics/ultralytics:latest

LABEL authors="Wenhui Zhou"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

RUN sed -i 's@http://archive.ubuntu.com/ubuntu/@https://mirrors.tuna.tsinghua.edu.cn/ubuntu/@g' /etc/apt/sources.list \
 && sed -i 's@http://security.ubuntu.com/ubuntu/@https://mirrors.tuna.tsinghua.edu.cn/ubuntu/@g' /etc/apt/sources.list

RUN apt-get update \
 && apt-get install -y --no-install-recommends wget gnupg ca-certificates \
 && wget -qO /tmp/cuda-keyring.deb \
       https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb \
 && dpkg -i /tmp/cuda-keyring.deb \
 && rm -f /tmp/cuda-keyring.deb \
 && rm -f /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      nvidia-cuda-toolkit \
      tensorrt=8.4.1.5-1+cuda11.6 \
      libnvinfer-dev=8.4.1-1+cuda11.6 \
      python3-pycuda \
 && apt-mark hold tensorrt libnvinfer8 libnvinfer-plugin8 libnvinfer-dev \
                libnvonnxparsers8 libnvparsers8 \
 && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --upgrade pip \
 && pip3 install --no-cache-dir \
      typing_extensions scipy tiff imagecodecs \
      scikit-learn scikit-image ptflops numpy==1.23.1

CMD ["/bin/bash"]
