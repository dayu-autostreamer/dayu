ARG REG=docker.io
FROM ${REG}/ultralytics/ultralytics:latest

LABEL authors="Wenhui Zhou"

RUN rm -rf /etc/apt/sources.list.d/ros-latest.list && \
    apt-get update && \
    pip3 install --upgrade pip && \
    pip3 install typing_extensions numpy==1.23.1 scipy tiff imagecodecs scikit-learn scikit-image -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN ["/bin/bash"]