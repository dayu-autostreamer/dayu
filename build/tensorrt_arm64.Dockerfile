ARG REG=docker.io
FROM ${REG}/ultralytics/ultralytics:latest-jetson-jetpack4

LABEL authors="Wenhui Zhou"

ENV DEBIAN_FRONTEND=noninteractive

RUN sed -i 's@http://archive.ubuntu.com/ubuntu/@https://mirrors.tuna.tsinghua.edu.cn/ubuntu/@g' /etc/apt/sources.list \
 && sed -i 's@http://security.ubuntu.com/ubuntu/@https://mirrors.tuna.tsinghua.edu.cn/ubuntu/@g' /etc/apt/sources.list \
 && sed -i '/cuda-internal.nvidia.com/d' /etc/apt/sources.list.d/*.list

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential python3-dev g++ \
      libsqlite3-dev libgeos-dev libproj-dev curl make \
      autoconf automake libtool pkg-config \
      git cmake  \
 && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /usr/src/gdal \
 && cd /usr/src/gdal \
 && git clone --depth 1 --branch v3.6.3 https://github.com/OSGeo/gdal.git \
 && cd gdal \
 && mkdir build \
 && cd build \
 && cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local \
 && make -j"$(nproc)" \
 && make install \
 && ldconfig \
 && rm -rf /usr/src/gdal

RUN pip3 install --upgrade pip \
 && pip3 install "setuptools<60.0.0" \
 && pip3 install \
      typing_extensions scipy tiff imagecodecs \
      scikit-learn scikit-image tensorrt pycuda \
      numpy==1.23.1 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN ["/bin/bash"]