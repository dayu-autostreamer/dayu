ARG REG=docker.io
FROM ${REG}/yuefan2022/tensorrt-ubuntu20.04-cuda11.6:latest

LABEL authors="Wenhui Zhou"

RUN rm -rf /etc/apt/sources.list.d/ros-latest.list && \
    apt-get update && \
    pip3 install --upgrade pip && \
    pip3 install opencv-python-headless typing_extensions  scipy tiff imagecodecs scikit-learn scikit-image -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip3 --default-timeout=1688 install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117 && \
    pip3 install tensorrt pycuda ptflops ultralytics numpy==1.23.1 -i https://pypi.tuna.tsinghua.edu.cn/simple && \

CMD ["/bin/bash"]
