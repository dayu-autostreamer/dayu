ARG REG=docker.io
FROM ${REG}/ultralytics/ultralytics:latest-jetson-jetpack4

LABEL authors="Wenhui Zhou"

ENV DEBIAN_FRONTEND=noninteractive

RUN sed -i 's@http://archive.ubuntu.com/ubuntu/@https://mirrors.tuna.tsinghua.edu.cn/ubuntu/@g' /etc/apt/sources.list && \
    sed -i 's@http://security.ubuntu.com/ubuntu/@https://mirrors.tuna.tsinghua.edu.cn/ubuntu/@g' /etc/apt/sources.list && \
    sed -i '/cuda-internal.nvidia.com/d' /etc/apt/sources.list.d/*.list && \
    apt-get clean && \
    apt-get update && \
    apt-get install -y python3-apt python3-distutils && \
    echo "deb https://ppa.launchpadcontent.net/ubuntugis/ppa/ubuntu bionic main" > /etc/apt/sources.list.d/ubuntugis-ppa.list && \
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 089EBE08314DF160 && \
    add-apt-repository -y ppa:ubuntugis/ppa && \
    apt-get update && \
    apt-get install -y software-properties-common && \
    apt-get install -y build-essential python3-dev g++ \
        libgdal-dev=3.6.3+dfsg-1~ubuntu22.04.1 \
        gdal-bin=3.6.3+dfsg-1~ubuntu22.04.1 && \
    pip3 install --upgrade pip && \
    pip3 install --upgrade "setuptools<60.0.0" && \
    pip3 install typing_extensions scipy tiff imagecodecs scikit-learn scikit-image tensorrt pycuda numpy==1.23.1 -i https://pypi.tuna.tsinghua.edu.cn/simple


RUN ["/bin/bash"]