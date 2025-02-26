ARG REG=docker.io
FROM ${REG}/diegofpsouza/numpy:0.0.1

LABEL authors="Wenhui Zhou"

ARG dependency_dir=dependency
ARG lib_dir=dependency/core/lib
ARG base_dir=dependency/core/controller
ARG code_dir=components/controller

ENV TZ=Asia/Shanghai

COPY ${lib_dir}/requirements.txt ./lib_requirements.txt
COPY ${base_dir}/requirements.txt ./base_requirements.txt

# install redis
RUN apt-get update && \
    apt-get install -y redis-server && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

 RUN echo "protected-mode no" >> /etc/redis/redis.conf && \
    echo "bind 0.0.0.0" >> /etc/redis/redis.conf


RUN pip3 install --upgrade pip && \
    pip install -r lib_requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install -r base_requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple


COPY ${dependency_dir} /home/dependency
ENV PYTHONPATH="/home/dependency"

WORKDIR /app
COPY  ${code_dir}/* /app/


CMD redis-server /etc/redis/redis.conf --daemonize yes && \
    gunicorn main:app -c ./gunicorn.conf.py


