ARG REG=docker.io
FROM ${REG}/ultralytics/ultralytics:latest-jetson-jetpack4

LABEL authors="Wenhui Zhou"

ENV DEBIAN_FRONTEND=noninteractive

RUN sed -i 's@http://archive.ubuntu.com/ubuntu/@https://mirrors.tuna.tsinghua.edu.cn/ubuntu/@g' /etc/apt/sources.list \
 && sed -i 's@http://security.ubuntu.com/ubuntu/@https://mirrors.tuna.tsinghua.edu.cn/ubuntu/@g' /etc/apt/sources.list \
 && sed -i '/cuda-internal.nvidia.com/d' /etc/apt/sources.list.d/*.list

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      apt-transport-https ca-certificates gnupg dirmngr \
      software-properties-common python3-apt python3-distutils \
 && rm -rf /var/lib/apt/lists/*


RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      gnupg dirmngr apt-transport-https ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys \
      FF0E7BBEC491C6A1 089EBE08314DF160 \
 && echo "deb https://ppa.launchpadcontent.net/ubuntugis/ppa/ubuntu bionic main" \
      > /etc/apt/sources.list.d/ubuntugis-ppa.list

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential python3-dev g++ \
      libgdal-dev gdal-bin \
 && rm -rf /var/lib/apt/lists/*


RUN pip3 install --upgrade pip \
 && pip3 install "setuptools<60.0.0" \
 && pip3 install \
      typing_extensions scipy tiff imagecodecs \
      scikit-learn scikit-image tensorrt pycuda \
      numpy==1.23.1 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple



RUN ["/bin/bash"]