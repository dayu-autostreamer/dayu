# Docker Containers for Dayu System

## introduction

Dayu is based on KubeEdge, which is a distributed cloud-edge system with docker containers. Thus, each component of our system is in the form of docker container.

To be noted, usually we take server with NVIDIA GPU for cloud (which is amd64 architecture) and take NVIDIA Jetson devices as edge (which are arm64 architecture). Therefore, building dockers should be carefully for architectures.

## Basic Source

Here list basic docker images used in the dayu system. Our own images can be built by dockerfiles in [this folder](../dockerfile).

**Tips**: If you have difficulty visiting [dockerhub](https://hub.docker.com/), constructing a private docker registry is recommended. Then you can replace `docker.io` as your registry address. **[Here](../../instructions/private_docker_registry.md) is a short instruction you can refer to.** 


```
# From others
docker.io/ultralytics/ultralytics:latest (amd64)
docker.io/ultralytics/ultralytics:latest-jetson-jetpack4 (arm64)
docker.io/redis:latest (amd64)


# Our own images
docker.io/dayuhub/tensorrt:trt8  (amd64/arm64)

docker.io/dayuhub/backend:{VERSION} (amd64)
docker.io/dayuhub/frontend:{VERSION} (amd64)
docker.io/dayuhub/datasource:{VERSION} (arm64)

docker.io/dayuhub/generator:{VERSION} (arm64)
docker.io/dayuhub/controller:{VERSION}  (amd64/arm64)
docker.io/dayuhub/dsitributor:{VERSION}  (amd64)
docker.io/dayuhub/scheduler:{VERSION}  (amd64)
docker.io/dayuhub/monitor:{VERSION}  (amd64/arm64)

docker.io/dayuhub/car-detection:{VERSION}  (amd64/arm64)
docker.io/dayuhub/face-detection:{VERSION}  (amd64/arm64)
docker.io/dayuhub/gender-classification:{VERSION}  (amd64/arm64)
docker.io/dayuhub/age-classification:{VERSION}  (amd64/arm64)

```


## How to build docker manifest list

You can refer to the [instruction](../../instructions/docker_build) to build dockers 

To make an image adaptive to different architectures (like amd64 and arm64), our docker images use docker manifest.

You can follow the instructions to build docker manifest.

- build different architecture images separately.

build amd64 image (or pull existed image) on amd64 device:
```bash
docker build ${registry}/${repository}/${image}:${tag-amd64}
docker push ${registry}/${repository}/${image}:${tag-amd64}
```

build arm64 image (or pull existed image) on arm64 device:
```bash
docker build ${registry}/${repository}/${image}:${tag-arm64}
docker push ${registry}/${repository}/${image}:${tag-arm64}
```

- build image manifest.
```bash
docker buildx imagetools create -t ${registry}/${repository}/${image}:${tag} ${registry}/${repository}/${image}:${tag-amd64} ${registry}/${repository}/${image}:${tag-arm64}
```

With docker manifest, you can pull docker image with the same tag on devices with different architectures. The system will get the correct image automatically.

## How to build different architectures on one device

You can you use docker buildx to build amd64 and arm64 images on amd64 computer. The instruction of setting docker buildx is [here](../../instructions/buildx.md)

## Easy way for building all images in our system

You can run [shell script](../../tools/build.sh) of building all required images.