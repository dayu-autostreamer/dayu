ARG REG=docker.io
FROM ${REG}/ultralytics/ultralytics:latest-jetson-jetpack5

LABEL authors="Wenhui Zhou"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

RUN sed -i 's@http://\(archive\|security\).ubuntu.com/ubuntu/@https://mirrors.tuna.tsinghua.edu.cn/ubuntu/@g' /etc/apt/sources.list; \
    if ls /etc/apt/sources.list.d/*.list >/dev/null 2>&1; then \
      sed -i '/cuda-internal.nvidia.com/d' /etc/apt/sources.list.d/*.list || true; \
    fi \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential python3-dev g++ curl make autoconf automake libtool pkg-config git \
      libsqlite3-dev libgeos-dev sqlite3 \
      libbz2-dev liblcms2-dev libsnappy-dev liblz4-dev libzopfli-dev \
      libopenjp2-7-dev libjpeg-turbo8-dev libjpeg-dev libpng-dev libwebp-dev libzstd-dev \
      libtiff-dev libgif-dev libjbig-dev liblzma-dev libcfitsio-dev libcharls-dev \
      libcurl4-gnutls-dev libexpat1-dev libxml2-dev zlib1g-dev libssl-dev \
      libde265-dev libx265-dev libavcodec-dev libavformat-dev libavutil-dev meson \
      libaec-dev libblosc-dev libbrotli-dev \
 && rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    cd /usr/src; \
    \
    # CMake 3.28.3
    wget https://github.com/Kitware/CMake/releases/download/v3.28.3/cmake-3.28.3.tar.gz; \
    tar xzf cmake-3.28.3.tar.gz; \
    cd cmake-3.28.3; ./bootstrap; make -j"$(nproc)"; make install; \
    cd /usr/src; rm -rf cmake-3.28.3*; \
    \
    # Brunsli
    git clone https://github.com/google/brunsli.git; \
    cd brunsli; git submodule update --init; mkdir build && cd build; \
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local ..; \
    make -j"$(nproc)"; make install; ldconfig; \
    cd /usr/src; rm -rf brunsli; \
    \
    # PROJ 9.1.1
    mkdir -p proj; \
    curl -L https://download.osgeo.org/proj/proj-9.1.1.tar.gz | tar -xz -C proj --strip-components=1; \
    cd proj; mkdir build && cd build; \
    cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local; \
    make -j"$(nproc)"; make install; ldconfig; \
    cd /usr/src; rm -rf proj; \
    \
    # GDAL 3.6.3
    mkdir -p gdal; \
    curl -L https://download.osgeo.org/gdal/3.6.3/gdal-3.6.3.tar.gz | tar -xz -C gdal --strip-components=1; \
    cd gdal; mkdir build && cd build; \
    cmake .. \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DPROJ_DIR=/usr/local \
      -DPROJ_INCLUDE_DIR=/usr/local/include \
      -DPROJ_LIBRARY=/usr/local/lib/libproj.so; \
    make -j"$(nproc)"; make install; ldconfig; \
    cd /usr/src; rm -rf gdal; \
    \
    # libdeflate 1.19
    wget https://github.com/ebiggers/libdeflate/archive/refs/tags/v1.19.tar.gz -O libdeflate.tar.gz; \
    mkdir libdeflate; tar -xzf libdeflate.tar.gz -C libdeflate --strip-components=1; \
    cd libdeflate; mkdir build && cd build; \
    cmake .. -DCMAKE_BUILD_TYPE=Release; \
    make -j"$(nproc)"; make install; ldconfig; \
    cd /usr/src; rm -rf libdeflate*; \
    \
    # libheif v1.14.0
    git clone https://github.com/strukturag/libheif.git libheif; \
    cd libheif; git checkout v1.14.0; mkdir build && cd build; \
    cmake .. \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DDETECT_LIBDE265=ON \
      -DDETECT_LIBX265=ON; \
    make -j"$(nproc)"; make install; ldconfig; \
    cd /usr/src; rm -rf libheif; \
    \
    # libtiff 4.5.0
    wget https://download.osgeo.org/libtiff/tiff-4.5.0.tar.gz; \
    tar xzf tiff-4.5.0.tar.gz; cd tiff-4.5.0; \
    ./configure --prefix=/usr/local --enable-zstd --with-zlib --with-webp; \
    make -j"$(nproc)"; make install; ldconfig; \
    rm -rf /usr/src/tiff-4.5.0*

ENV MAX_JOBS=1

RUN  apt-get update \
  && apt-get remove -y python3-yaml python3-psutil \
  && pip3 install --upgrade pip \
  && pip3 install "setuptools<60.0.0" \
 && pip3 install --no-cache-dir tiff \
 && pip3 install --no-cache-dir imagecodecs \
      --global-option="build_ext" \
      --global-option="--skip-jpeg8" \
      --global-option="--skip-jpegls" \
      --global-option="--skip-jpegxl" \
      --global-option="--skip-jpegxr" \
      --global-option="--skip-lz4" \
      --global-option="--skip-zfp" \
 && pip3 install --no-cache-dir \
      typing_extensions scipy scikit-learn scikit-image \
      tensorrt pycuda \
      ptflops==0.7.2.2 \
 && pip3 install --no-build-isolation torch-scatter==2.1.2 \
 && pip3 install --no-build-isolation torch-sparse==0.6.18 \
 && pip3 install --no-build-isolation torch-geometric==2.6.1

RUN pip3 uninstall -y numpy \
 && pip3 install --no-cache-dir  "numpy==1.23.5"

CMD ["/bin/bash"]
