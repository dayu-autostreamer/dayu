ARG REG=docker.io
FROM ${REG}/ultralytics/ultralytics:latest

LABEL authors="Wenhui Zhou"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

RUN sed -i \
      -e 's@http://archive.ubuntu.com/ubuntu/@https://mirrors.tuna.tsinghua.edu.cn/ubuntu/@g' \
      -e 's@http://security.ubuntu.com/ubuntu/@https://mirrors.tuna.tsinghua.edu.cn/ubuntu/@g' \
    /etc/apt/sources.list

RUN apt-get update \
 && apt-get install -y --no-install-recommends wget gnupg ca-certificates

RUN wget -qO /tmp/cuda-keyring.deb \
      https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb \
 && dpkg -i /tmp/cuda-keyring.deb \
 && rm -f /tmp/cuda-keyring.deb

RUN wget -qO /tmp/nvidia-ml-repo.deb \
      https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/nvidia-machine-learning-repo-ubuntu2004_1.0.0-1_amd64.deb \
 && dpkg -i /tmp/nvidia-ml-repo.deb \
 && rm -f /tmp/nvidia-ml-repo.deb

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
      nvidia-cuda-toolkit \
      libnvinfer8=8.4.1-1+cuda11.6 \
      libnvinfer-plugin8=8.4.1-1+cuda11.6 \
      libnvonnxparsers8=8.4.1-1+cuda11.6 \
      libnvparsers8=8.4.1-1+cuda11.6 \
      tensorrt=8.4.1.5-1+cuda11.6 \
      python3-libnvinfer=8.4.1-1+cuda11.6 \
      python3-pycuda \
 && apt-mark hold \
      libnvinfer8 libnvinfer-plugin8 libnvonnxparsers8 libnvparsers8 \
      python3-libnvinfer tensorrt \
 && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --upgrade pip \
 && pip3 install --no-cache-dir \
      typing_extensions scipy tiff imagecodecs \
      scikit-learn scikit-image ptflops numpy==1.23.1

CMD ["/bin/bash"]