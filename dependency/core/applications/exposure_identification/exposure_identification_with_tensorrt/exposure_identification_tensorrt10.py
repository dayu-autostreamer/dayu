import cv2
import numpy as np
from typing import List

import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

from core.lib.common import Context, LOGGER


class ExposureIdentificationTensorRT10:
    def __init__(self, weights, device=0):
        self.weights = weights

        self.ctx = cuda.Device(device).make_context()
        stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)

        engine_file_path = self.weights
        with open(engine_file_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        input_binding_names = []
        output_binding_names = []

        for binding_name in engine:
            shape = engine.get_tensor_shape(binding_name)
            size = trt.volume(shape)
            dtype = trt.nptype(engine.get_tensor_dtype(binding_name))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append to the appropriate list.
            if engine.get_tensor_mode(binding_name) == trt.TensorIOMode.INPUT:
                input_binding_names.append(binding_name)
                self.input_w = shape[-1]
                self.input_h = shape[-2]
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            elif engine.get_tensor_mode(binding_name) == trt.TensorIOMode.OUTPUT:
                output_binding_names.append(binding_name)
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)
            else:
                # Unknown tensor mode, skip it
                pass

        self.stream = stream
        self.context = context
        self.engine = engine
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.input_binding_names = input_binding_names
        self.output_binding_names = output_binding_names
        self.batch_size = engine.get_tensor_shape(input_binding_names[0])[0]

        self.warm_up_turns = 5

        self.classes = ['Drawing', 'Hentai', 'Neutral', 'Porn', 'Sexy']

        self.warm_up()

    def __del__(self):
        self.destroy()

    def destroy(self):
        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()

    def preprocess_image(self, raw_bgr_image):
        """
        description: Read an image from image path, resize and pad it to target size,
                     normalize to [0,1],transform to NCHW format.
        param:
            input_image_path: str, image path
        return:
            image:  the processed image
            image_raw: the original image
            h: original height
            w: original width
        """
        image_rgb = cv2.cvtColor(raw_bgr_image, cv2.COLOR_BGR2RGB)

        image_resized = cv2.resize(image_rgb, (self.input_w, self.input_h),
                                   interpolation=cv2.INTER_LINEAR)

        image = image_resized.astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std

        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0)
        image = np.ascontiguousarray(image)

        return image, raw_bgr_image, image_resized.shape[0], image_resized.shape[1]

    def warm_up(self):
        LOGGER.info('Warming up...')
        for i in range(self.warm_up_turns):
            im = np.zeros([self.input_h, self.input_w, 3], dtype=np.uint8)
            self.infer(im)
            del im

    def infer(self, raw_image):
        # Make self the active context, pushing it on top of the context stack.
        self.ctx.push()

        # Restore
        stream = self.stream
        context = self.context
        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        input_binding_names = self.input_binding_names
        output_binding_names = self.output_binding_names
        # Do image preprocess
        input_image, image_raw, origin_h, origin_w = self.preprocess_image(raw_image)

        # Copy input image to host buffer
        np.copyto(host_inputs[0], input_image.ravel())
        # Transfer input data  to the GPU.
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        # Run inference.
        context.set_tensor_address(input_binding_names[0], cuda_inputs[0])
        context.set_tensor_address(output_binding_names[0], cuda_outputs[0])
        context.execute_async_v3(stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        # Synchronize the stream
        stream.synchronize()
        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()
        # Here we use the first row of output
        output = host_outputs[0]
        # Reshape output to handle batch dimension
        # Output shape should be [batch_size, num_classes] when flattened
        num_classes = len(self.classes)
        output = output.reshape(self.batch_size, -1)[0]
        
        # Ensure we only use the first num_classes elements
        if len(output) > num_classes:
            output = output[:num_classes]
        
        age_output = np.argmax(output)
        
        # Ensure index is within valid range
        if age_output >= len(self.classes):
            LOGGER.warning(f'Output index {age_output} out of range (max: {len(self.classes)-1}), using 0')
            age_output = 0

        return self.classes[age_output]
