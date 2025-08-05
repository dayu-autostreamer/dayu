class FlopsEstimator:
    def __init__(self, model, input_shape: tuple):
        self.model = model
        self.input_shape = input_shape

    def compute_flops(self):
        # Try model.info() first for yolo models
        try:
            if hasattr(self.model, 'info') and callable(getattr(self.model, 'info')):
                info = self.model.info()  # (layers, params, grads, gflops)
                flops = info[3] * 1e9
                return flops
        except Exception as e:
            pass
            
        # Use ptflop instead
        import ptflops
        macs, _ = ptflops.get_model_complexity_info(
            self.model,
            self.input_shape,
            print_per_layer_stat=False,
            as_strings=False,
            verbose=False
        )
        flops = 2 * macs
        return flops
