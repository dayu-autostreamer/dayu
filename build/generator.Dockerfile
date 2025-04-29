ARG REG=docker.io
FROM ${REG}/dayuhub/tensorrt:trt8


LABEL authors="Wenhui Zhou"

ARG dependency_dir=dependency
ARG lib_dir=dependency/core/lib
ARG base_dir=dependency/core/generator
ARG code_dir=components/generator

# Required to build Ubuntu 20.04 without user prompts with DLFW container
ENV DEBIAN_FRONTEND=noninteractive

ENV TZ=Asia/Shanghai

COPY ${lib_dir}/requirements.txt ./lib_requirements.txt
COPY ${base_dir}/requirements.txt ./base_requirements.txt

ENV LANG=en_US.UTF-8

# scikit-image
RUN apt-get update && \
    apt-get remove -y python3-yaml && \
    pip3 install --upgrade pip && \
    apt-get install -y python3-sklearn && \
    apt-get install -y --no-install-recommends libaec-dev libblosc-dev libffi-dev libbrotli-dev libboost-all-dev libbz2-dev && \
    apt-get install -y --no-install-recommends libgif-dev libopenjp2-7-dev liblcms2-dev libjpeg-dev libjxr-dev liblz4-dev liblzma-dev libpng-dev libsnappy-dev libwebp-dev libzopfli-dev libzstd-dev && \
    pip3 install scikit-image  && \
#
    pip3 install -r lib_requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip3 install -r base_requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple



COPY ${dependency_dir} /home/dependency
ENV PYTHONPATH="/home/dependency"

WORKDIR /app
COPY  ${code_dir}/* /app/

CMD ["python3", "main.py"]