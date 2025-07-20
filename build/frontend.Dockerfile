ARG REG=docker.io
FROM ${REG}/node:20

LABEL authors="Skyrim"

ARG code_dir=frontend

ENV TimeZone=Asia/Shanghai

COPY  ${code_dir}/ /app/
WORKDIR /app

RUN npm install -g cnpm --registry=https://registry.npmmirror.com/ && \
    cnpm install

CMD ["cnpm", "run", "dev"]
