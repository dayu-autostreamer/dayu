ARG REG=docker.io
FROM ${REG}/dayuhub/tensorrt:jp6

LABEL authors="Haoyang Su"

ARG dependency_dir=dependency
ARG lib_dir=dependency/core/lib
ARG base_dir=dependency/core/processor
ARG code_dir=components/processor
ARG app_dir=dependency/core/applications/pedestrian_detection

ENV TZ=Asia/Shanghai

COPY ${lib_dir}/requirements.txt ./lib_requirements.txt
COPY ${base_dir}/requirements.txt ./base_requirements.txt
COPY ${app_dir}/requirements.txt ./app_requirements.txt


RUN pip3 install --upgrade pip && \
    pip3 install -r lib_requirements.txt --ignore-installed -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip3 install -r base_requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip3 install -r app_requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip3 uninstall -y numpy \
 && pip3 install --no-cache-dir  "numpy==1.23.5"

COPY ${dependency_dir} /home/dependency
ENV PYTHONPATH="/home/dependency"

WORKDIR /app
COPY  ${code_dir}/* /app/

CMD ["python3", "-m", "gunicorn", "main:app", "-c", "./gunicorn.conf.py"]