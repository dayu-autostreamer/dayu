class FlopsEstimator:
    def __init__(self, model, input_shape: tuple):
        self.model = model
        self.input_shape = input_shape

    def compute_flops(self):
        import ptflops
        macs, _ = ptflops.get_model_complexity_info(
            self.model,
            self.input_shape,
            print_per_layer_stat=False,
            verbose=False
        )
        flops = 2 * macs

        return flops
