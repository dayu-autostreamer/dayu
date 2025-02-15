ARG REG=docker.io
# TODO: 修改镜像tag，tx2要用:latest-jetson-jetpack4
FROM ${REG}/ultralytics/ultralytics:latest-arm64

LABEL authors="Wenyi Dai"

ARG dependency_dir=dependency
ARG lib_dir=dependency/core/lib
ARG base_dir=dependency/core/processor
ARG code_dir=components/processor

# 修改
ARG app_dir=dependency/core/applications/model_switch_detection

ENV TZ=Asia/Shanghai

COPY ${lib_dir}/requirements.txt ./lib_requirements.txt
COPY ${base_dir}/requirements.txt ./base_requirements.txt

# 修改，添加两个requirements
COPY ${app_dir}/requirements_arm64.txt ./app_requirements.txt

# 修改 pip 安装命令，添加 --use-pep517 参数
RUN pip3 install --upgrade pip && \
    pip3 install --use-pep517 -r lib_requirements.txt --ignore-installed -i https://mirrors.aliyun.com/pypi/simple && \
    pip3 install -r base_requirements.txt -i https://mirrors.aliyun.com/pypi/simple && \
    pip3 install -r app_requirements.txt -i https://mirrors.aliyun.com/pypi/simple

COPY ${dependency_dir} /home/dependency
ENV PYTHONPATH="/home/dependency"

# new add for docker image build
ENV PROCESSOR_NAME="detector_processor"
ENV SERVICE_NAME="processor-model-switch-detection"
ENV DETECTOR_PARAMETERS="{'key':'value'}"
ENV PRO_QUEUE_NAME="simple"
ENV NAMESPACE="aaa"
ENV KUBERNETES_SERVICE_HOST="xxx"
ENV KUBERNETES_SERVICE_PORT="xxx"

WORKDIR /app
COPY  ${code_dir}/* /app/

# 修改，bin/bash
# CMD ["gunicorn", "main:app", "-c", "./gunicorn.conf.py"]
CMD ["/bin/bash"]

# 进入后跑main函数