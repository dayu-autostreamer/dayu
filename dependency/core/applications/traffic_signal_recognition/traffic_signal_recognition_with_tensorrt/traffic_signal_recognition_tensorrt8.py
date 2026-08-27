class TrafficSignalRecognitionTensorRT8:
    def __init__(self, weights='', plugin_library='', device=0):
        self.weights = weights
        self.plugin_library = plugin_library
        self.device = device
        raise NotImplementedError('traffic-signal-recognition TensorRT implementation is not implemented yet.')
