## Module Integration Notes

### Application Module Development
1. Create a new project under `dependency/core/applications`. Use `car_detection` as a reference. Make sure `__init__.py` exports the module.
   The core is to implement a class with a `__call__` method to process a batch of input images, and pay attention to input/output types.
2. Local testing method: write a `main` function for the application and test whether calling `__call__` works correctly.

#### Common pitfalls

1. If the module has submodules, package imports may fail in different execution contexts. One simple (but practical) approach:
    ```python
    def _import_random_switch_module():
        if __name__ == '__main__':
            from switch_module.random_switch import RandomSwitch
        else:
            from .switch_module.random_switch import RandomSwitch
        return RandomSwitch
    ```
    This can handle different contexts.

2. Adding modules via `sys.path` can cause imports of packages with the same name.
   For example, two detection projects may both have a `models` module; this can lead to confusion.
   It's best to distinguish them during initialization and avoid importing identically named modules from multiple projects at the same time:
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

3. When testing Docker image builds, you may not be able to pass parameters when instantiating the object
   (the application will eventually be started in k8s via YAML).
   You can write a parameterless test class that inherits the parameterized class, then modify the app root `__init__.py`
   to export the parameterless test class:
    ```
    from .detection_wrapper import ModelSwitchDetection as Detector
    # from .detection_wrapper import ModelSwitchDetectionTestYolo as Detector
    # from .detection_wrapper import ModelSwitchDetectionTestOfa as Detector
    __all__ = ["Detector"]
    ```

### Build Docker Images
1. Create the application's Dockerfile under the `build` directory.
   Note that you need to distinguish architectures (amd/arm) because the base images and build flows differ.
2. A template for local image build testing:
    ```yaml
    ARG REG=docker.io
    # TODO: update the base image tag
    FROM ${REG}/ultralytics/ultralytics:latest-arm64

    LABEL authors="Wenyi Dai"

    ARG dependency_dir=dependency
    ARG lib_dir=dependency/core/lib
    ARG base_dir=dependency/core/processor
    ARG code_dir=components/processor

    # Change to your own application directory
    ARG app_dir=dependency/core/applications/model_switch_detection

    ENV TZ=Asia/Shanghai

    COPY ${lib_dir}/requirements.txt ./lib_requirements.txt
    COPY ${base_dir}/requirements.txt ./base_requirements.txt

    # Change: add two requirements files (arm/amd)
    COPY ${app_dir}/requirements_arm64.txt ./app_requirements.txt

    # Change: add --use-pep517 to pip install commands
    RUN pip3 install --upgrade pip && \
        pip3 install --use-pep517 -r lib_requirements.txt --ignore-installed -i https://mirrors.aliyun.com/pypi/simple && \
        pip3 install -r base_requirements.txt -i https://mirrors.aliyun.com/pypi/simple && \
        pip3 install -r app_requirements.txt -i https://mirrors.aliyun.com/pypi/simple

    COPY ${dependency_dir} /home/dependency
    ENV PYTHONPATH="/home/dependency"

    # Note: SERVICE_NAME should be 'processor-{your-app-name}', and replace '_' with '-'
    ENV PROCESSOR_NAME="detector_processor"
    ENV SERVICE_NAME="processor-model-switch-detection"
    ENV DETECTOR_PARAMETERS="{'key':'value'}"
    ENV PRO_QUEUE_NAME="simple"
    ENV NAMESPACE="aaa"
    ENV KUBERNETES_SERVICE_HOST="xxx"
    ENV KUBERNETES_SERVICE_PORT="xxx"

    WORKDIR /app
    COPY  ${code_dir}/* /app/

    # Change: /bin/bash
    # CMD ["gunicorn", "main:app", "-c", "./gunicorn.conf.py"]
    CMD ["/bin/bash"]

    # After entering the container, run the main function
    ```
3. Run (example):
    ```bash
    docker build -t dayu-test -f build/model_switch_detection_arm64.Dockerfile .
    docker run --rm -it -v $(pwd)/ofa_weights:/ofa_weights dayu-test
    ```
    Use `-v` to mount the weights file for testing. Make sure the paths inside Docker and in the parameterless test class match.
    After entering, run Python:
    ```python
    from core.processor import ProcessorServer
    app = ProcessorServer().app
    ```
    Verify that the processor initializes successfully.

### Other
1. Make sure the supernet weights match the specific detection classes in the TorchVision version inside the Docker image,
   otherwise you need to convert the network.
   For example, the RPN head structure of Faster R-CNN may have minor changes.
   Older versions do not have `Conv2dNormActivation`; you need to extract the Conv from `Conv2dNormActivation` and save weights again.
