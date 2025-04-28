ARG REG=docker.io
FROM  ${REG}/nvidia/l4t-pytorch:r32.7.1-pth1.10-py3

LABEL authors="Wenhui Zhou"

ENV DEBIAN_FRONTEND=noninteractive

COPY pdk_files /pdk_files

# Install requried libraries
RUN apt-key adv --recv-keys --keyserver keyserver.ubuntu.com 16FAAD7AF99A65E2 && \
    apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:ubuntu-toolchain-r/test

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libcurl4-openssl-dev \
        wget \
        git \
        pkg-config \
        sudo \
        ssh \
        libssl-dev \
        pbzip2 \
        pv \
        bzip2 \
        unzip \
        devscripts \
        lintian \
        fakeroot \
        dh-make \
        build-essential \
        python3-pip \
        libopenblas-base \
        libopenmpi-dev \
        libomp-dev \
        libjpeg-dev \
        zlib1g-dev \
        libpython3-dev \
        libopenblas-dev \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev

RUN apt-get --purge remove  cuda*
RUN rm -rf /usr/local/cuda*

RUN dpkg -i /pdk_files/cuda-repo-l4t-10-2-local_10.2.460-1_arm64.deb
RUN apt-key add /var/cuda-repo*/*.pub \
    && apt-get -y update \
    && apt-get -y install -f cuda-cudart-dev-10-2 \
    && apt-get -y install -f cuda-toolkit-10-2

RUN     dpkg -i /pdk_files/libcudnn8_8.2.1.32-1+cuda10.2_arm64.deb \
     && dpkg -i /pdk_files/libcudnn8-dev_8.2.1.32-1+cuda10.2_arm64.deb \
     && dpkg -i /pdk_files/libcudnn8-samples_8.2.1.32-1+cuda10.2_arm64.deb \
     && dpkg -i /pdk_files/libnvinfer8_8.2.1-1+cuda10.2_arm64.deb  \
     && dpkg -i /pdk_files/libnvinfer-dev_8.2.1-1+cuda10.2_arm64.deb \
     && dpkg -i /pdk_files/libnvparsers8_8.2.1-1+cuda10.2_arm64.deb   \
     && dpkg -i /pdk_files/libnvparsers-dev_8.2.1-1+cuda10.2_arm64.deb \
     && dpkg -i /pdk_files/libnvinfer-plugin8_8.2.1-1+cuda10.2_arm64.deb \
     && dpkg -i /pdk_files/libnvinfer-plugin-dev_8.2.1-1+cuda10.2_arm64.deb \
     && dpkg -i /pdk_files/libnvonnxparsers8_8.2.1-1+cuda10.2_arm64.deb \
     && dpkg -i /pdk_files/libnvonnxparsers-dev_8.2.1-1+cuda10.2_arm64.deb \
     && dpkg -i /pdk_files/libnvinfer-bin_8.2.1-1+cuda10.2_arm64.deb \
     && dpkg -i /pdk_files/libnvinfer-samples_8.2.1-1+cuda10.2_all.deb  \
     && dpkg -i /pdk_files/libnvinfer-doc_8.2.1-1+cuda10.2_all.deb \
     && dpkg -i /pdk_files/graphsurgeon-tf_8.2.1-1+cuda10.2_arm64.deb \
     && dpkg -i /pdk_files/uff-converter-tf_8.2.1-1+cuda10.2_arm64.deb \
     && dpkg -i /pdk_files/python3-libnvinfer_8.2.1-1+cuda10.2_arm64.deb \
     && dpkg -i /pdk_files/python3-libnvinfer-dev_8.2.1-1+cuda10.2_arm64.deb \
     && dpkg -i /pdk_files/tensorrt_8.2.1.9-1+cuda10.2_arm64.deb

RUN apt-get -y update && apt-get -y -f install \
     && apt-get install -y  /pdk_files/OpenCV-4.1.1-2-gd5a58aa75-aarch64-libs.deb \
     && apt-get install -y /pdk_files/OpenCV-4.1.1-2-gd5a58aa75-aarch64-dev.deb \
     && apt-get install -y /pdk_files/OpenCV-4.1.1-2-gd5a58aa75-aarch64-samples.deb  \
     && apt-get install -y /pdk_files/OpenCV-4.1.1-2-gd5a58aa75-aarch64-licenses.deb \
     && apt-get install -y /pdk_files/OpenCV-4.1.1-2-gd5a58aa75-aarch64-python.deb


WORKDIR /

RUN rm -rf /pdk_files


RUN ["/bin/bash"]