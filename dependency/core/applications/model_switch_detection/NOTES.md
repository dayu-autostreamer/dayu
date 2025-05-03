## 模块集成记录

### 应用模块开发
1. `dependency/core/applications`目录下新建项目，代码参考car_detection，注意`__init__.py`导出模块。核心是编写一个包含`__call__`方法的类用以处理一组输入图像，注意输入输出类型。
2. 本地测试方法：为应用编写主函数测试`__call__`方法调用是否正常。

#### 一些踩坑点：

1. 如果模块有子模块，包的导入可能会有问题，一种简单粗暴可行的写法：
    ```python
    def _import_random_switch_module():
        if __name__ == '__main__':
            from switch_module.random_switch import RandomSwitch
        else:
            from .switch_module.random_switch import RandomSwitch
        return RandomSwitch
    ```
    可以分别处理不同的context。

2. 用`sys.path`添加模块可能会有重名包的导入，比如两个目标检测的项目实现都有`models`模块，这种情况会导致混乱，所以最好在初始化参数中做区分，避免同时导入不同项目中的同名模块：
    ```python
    if model_type == 'yolo':
        YoloInference = _import_yolo_inference_module()
        self.detector = YoloInference(*args, **kwargs)
    elif model_type == 'ofa':
        OfaInference = _import_ofa_inference_module()
        self.detector = OfaInference(*args, **kwargs)

    else:
        raise ValueError('Invalid type')
    ```

3. 测试docker镜像构建时，实例化对象时无法传参（应用最后会在k8s中通过yaml文件启动），可以写无参初始化的测试类继承有参的类，再修改应用根目录下的`__init__.py`导出待测试的无参测试类：
    ```
    from .detection_wrapper import ModelSwitchDetection as Detector
    # from .detection_wrapper import ModelSwitchDetectionTestYolo as Detector
    # from .detection_wrapper import ModelSwitchDetectionTestOfa as Detector
    __all__ = ["Detector"]
    ```

### 构建docker镜像
1. `build`目录下新建应用的Dockerfile，注意需要区分架构（amd/arm），涉及到不同的基础镜像，构建流程不同。
2. 本地测试镜像构建的一个模板：
    ```yaml
    ARG REG=docker.io
    # TODO: 修改基础镜像tag
    FROM ${REG}/ultralytics/ultralytics:latest-arm64

    LABEL authors="Wenyi Dai"

    ARG dependency_dir=dependency
    ARG lib_dir=dependency/core/lib
    ARG base_dir=dependency/core/processor
    ARG code_dir=components/processor

    # 修改为自己的应用目录
    ARG app_dir=dependency/core/applications/model_switch_detection

    ENV TZ=Asia/Shanghai

    COPY ${lib_dir}/requirements.txt ./lib_requirements.txt
    COPY ${base_dir}/requirements.txt ./base_requirements.txt

    # 修改，添加两个requirements（arm/amd）
    COPY ${app_dir}/requirements_arm64.txt ./app_requirements.txt

    # 修改 pip 安装命令，添加 --use-pep517 参数
    RUN pip3 install --upgrade pip && \
        pip3 install --use-pep517 -r lib_requirements.txt --ignore-installed -i https://mirrors.aliyun.com/pypi/simple && \
        pip3 install -r base_requirements.txt -i https://mirrors.aliyun.com/pypi/simple && \
        pip3 install -r app_requirements.txt -i https://mirrors.aliyun.com/pypi/simple

    COPY ${dependency_dir} /home/dependency
    ENV PYTHONPATH="/home/dependency"

    # 注意SERVICE_NAME的格式为'processor-{你的应用名称}'，应用名称的'_'改为'-'
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
    ```
3. 运行，一个示例：
    ```bash
    docker build -t dayu-test -f build/model_switch_detection_arm64.Dockerfile .
    docker run --rm -it -v $(pwd)/ofa_weights:/ofa_weights dayu-test
    ```
    用`-v`把权重文件做映射进行测试，注意docker中的路径和无参测试类中的路径保持一致。进入后运行python：
    ```python
    from core.processor import ProcessorServer
    app = ProcessorServer().app
    ```
    测试processor类是否初始化成功。

### 其他
1. 一定要注意supernet的权重要和docker镜像中torch vision的具体detection类匹配，否则需要做网络转换，比如Faster RCNN的RPN head结构有微小变化，旧版本没有Conv2dNormActivation实现，要把Conv2dNormActivation里的Conv抠出来重新保存权重。