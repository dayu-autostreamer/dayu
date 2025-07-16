ARG REG=docker.io
FROM  ${REG}/ultralytics/ultralytics:latest-jetson-jetpack4

LABEL authors="Wenhui Zhou"

ENV DEBIAN_FRONTEND=noninteractive

RUN sed -i 's@http://archive.ubuntu.com/ubuntu/@https://mirrors.tuna.tsinghua.edu.cn/ubuntu/@g' /etc/apt/sources.list && \
    sed -i 's@http://security.ubuntu.com/ubuntu/@https://mirrors.tuna.tsinghua.edu.cn/ubuntu/@g' /etc/apt/sources.list && \
    sed -i '/cuda-internal.nvidia.com/d' /etc/apt/sources.list.d/*.list && \
    apt-get clean && \
    apt-get update && \
    apt-get install -y build-essential python3-dev g++ && \
    pip3 install --upgrade pip && \
    pip3 install typing_extensions scipy tiff imagecodecs scikit-learn scikit-image tensorrt pycuda numpy==1.23.1 -i https://pypi.tuna.tsinghua.edu.cn/simple


RUN ["/bin/bash"]