ARG REG=docker.io
FROM ${REG}/ultralytics/ultralytics:latest

LABEL authors="Wenhui Zhou"

RUN apt-get update && \
    apt-get install -y build-essential python3-dev g++ && \
    pip3 install --upgrade pip && \
    pip3 install typing_extensions scipy tiff imagecodecs scikit-learn scikit-image tensorrt pycuda numpy==1.23.1 -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN ["/bin/bash"]