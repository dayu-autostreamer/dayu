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
      libsqlite3-dev libgeos-dev curl make \
      autoconf automake libtool pkg-config \
      git cmake \
      libcurl4-gnutls-dev libexpat1-dev libxml2-dev \
      zlib1g-dev libssl-dev libpng-dev libjpeg-dev \
      libtiff-dev libwebp-dev libzstd-dev \
      sqlite3 libsqlite3-dev \
      libaec-dev \
      libblosc-dev libbrotli-dev \
 && rm -rf /var/lib/apt/lists/*

RUN cd /usr/src \
 && wget https://github.com/Kitware/CMake/releases/download/v3.28.3/cmake-3.28.3.tar.gz \
 && tar -xzf cmake-3.28.3.tar.gz \
 && cd cmake-3.28.3 \
 && ./bootstrap \
 && make -j"$(nproc)" \
 && make install \
 && rm -rf /usr/src/cmake-3.28.3*

RUN cd /usr/src \
 && git clone https://github.com/google/brunsli.git \
 && cd brunsli \
 && git submodule update --init \
 && mkdir build \
 && cd build \
 && cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local .. \
 && make -j"$(nproc)" \
 && make install \
 && ldconfig \
 && rm -rf /usr/src/brunsli

RUN mkdir -p /usr/src/proj \
 && cd /usr/src/proj \
 && curl -L https://download.osgeo.org/proj/proj-9.1.1.tar.gz | tar -xz \
 && cd proj-9.1.1 \
 && mkdir build \
 && cd build \
 && cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local \
 && make -j"$(nproc)" \
 && make install \
 && ldconfig \
 && rm -rf /usr/src/proj

RUN mkdir -p /usr/src/gdal \
 && cd /usr/src/gdal \
 && curl -L https://download.osgeo.org/gdal/3.6.3/gdal-3.6.3.tar.gz | tar -xz \
 && cd gdal-3.6.3 \
 && mkdir build \
 && cd build \
 && cmake .. \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DPROJ_DIR=/usr/local \
    -DPROJ_INCLUDE_DIR=/usr/local/include \
    -DPROJ_LIBRARY=/usr/local/lib/libproj.so \
 && make -j"$(nproc)" \
 && make install \
 && ldconfig \
 && rm -rf /usr/src/gdal

RUN apt-get update && \
    apt-get install -y --no-install-recommends libbz2-dev liblcms2-dev \
           libsnappy-dev liblz4-dev libzopfli-dev libopenjp2-7-dev \
           libjpeg-turbo8-dev zlib1g-dev

RUN cd /usr/src \
 && wget https://github.com/ebiggers/libdeflate/archive/refs/tags/v1.19.tar.gz -O libdeflate.tar.gz \
 && mkdir libdeflate \
 && tar -xzf libdeflate.tar.gz -C libdeflate --strip-components=1 \
 && cd libdeflate \
 && mkdir build \
 && cd build \
 && cmake .. -DCMAKE_BUILD_TYPE=Release \
 && make -j"$(nproc)" \
 && make install \
 && ldconfig \
 && cd / \
 && rm -rf /usr/src/libdeflate*

RUN apt-get update && apt-get install -y --no-install-recommends \
      cmake meson git autoconf automake libde265-dev libx265-dev \
      libavcodec-dev libavformat-dev libavutil-dev \
 && git clone https://github.com/strukturag/libheif.git /usr/src/libheif \
 && cd /usr/src/libheif \
 && git checkout v1.14.0 \
 && mkdir build && cd build \
 && cmake .. -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_INSTALL_PREFIX=/usr/local \
            -DDETECT_LIBDE265=ON \
            -DDETECT_LIBX265=ON \
 && make -j"$(nproc)" && make install \
 && ldconfig \
 && rm -rf /usr/src/libheif

RUN apt-get update && \
    apt-get install -y --no-install-recommends  libgif-dev  \
    libjbig-dev liblzma-dev  libcfitsio-dev  libcharls-dev

RUN pip3 install --upgrade pip \
 && pip3 install "setuptools<60.0.0" \
 && pip3 install --no-cache-dir imagecodecs \
      --global-option="build_ext" \
      --global-option="--skip-jpeg8" \
      --global-option="--skip-jpegls" \
      --global-option="--skip-jpegxl" \
      --global-option="--skip-jpegxr" \
      --global-option="--skip-lz4" \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip3 install --upgrade pip \
 && pip install typing_extensions scipy tiff  \
      scikit-learn scikit-image tensorrt pycuda \
      numpy==1.23.1 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN ["/bin/bash"]