ARG REG=docker.io
FROM ${REG}/dayuhub/tensorrt:trt8


LABEL authors="Haoyang Su"

ENV DEBIAN_FRONTEND=noninteractive

ARG dependency_dir=dependency
ARG lib_dir=dependency/core/lib
ARG base_dir=dependency/core/processor
ARG code_dir=components/processor
ARG app_dir=dependency/core/applications/exposure_identification

RUN apt-get update && \
    apt-get install -y tzdata python3-pip

ENV TZ=Asia/Shanghai

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && \
    echo $TZ > /etc/timezone

COPY ${lib_dir}/requirements.txt ./lib_requirements.txt
COPY ${base_dir}/requirements.txt ./base_requirements.txt
COPY ${app_dir}/requirements_arm64.txt ./app_requirements.txt

RUN pip3 install --upgrade pip && \
    pip3 install -r lib_requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip3 install -r base_requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip3 install -r app_requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY ${dependency_dir} /home/dependency
ENV PYTHONPATH="/home/dependency"

WORKDIR /app
COPY  ${code_dir}/* /app/

CMD ["gunicorn", "main:app", "-c", "./gunicorn.conf.py"]