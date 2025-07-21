ARG REG=docker.io
FROM ${REG}/dayuhub/tensorrt:trt8

LABEL authors="Wenyi Dai"

ARG dependency_dir=dependency
ARG lib_dir=dependency/core/lib
ARG base_dir=dependency/core/processor
ARG code_dir=components/processor
ARG app_dir=dependency/core/applications/model_switch_detection

ENV TZ=Asia/Shanghai

COPY ${lib_dir}/requirements.txt ./lib_requirements.txt
COPY ${base_dir}/requirements.txt ./base_requirements.txt
COPY ${app_dir}/requirements.txt ./app_requirements.txt

RUN apt-get update && \
    apt-get install -y build-essential python3-dev && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip && \
    pip3 install --use-pep517 -r lib_requirements.txt --ignore-installed -i https://mirrors.aliyun.com/pypi/simple && \
    pip3 install -r base_requirements.txt -i https://mirrors.aliyun.com/pypi/simple && \
    pip3 install -r app_requirements.txt -i https://mirrors.aliyun.com/pypi/simple

COPY ${dependency_dir} /home/dependency
ENV PYTHONPATH="/home/dependency"

WORKDIR /app
COPY  ${code_dir}/* /app/

CMD ["python3", "-m", "gunicorn", "main:app", "-c", "./gunicorn.conf.py"]
