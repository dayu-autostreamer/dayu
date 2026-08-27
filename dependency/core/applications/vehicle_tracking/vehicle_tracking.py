from core.lib.common import Context, LOGGER


class VehicleTracking:
    def __init__(self, trt_weights='', trt_plugin_library='', non_trt_weights='', device=0,
                 high_score_threshold=0.35, low_score_threshold=0.10, new_track_threshold=0.35,
                 match_score_threshold=0.35, secondary_iou_threshold=0.25, max_age=8):
        use_tensorrt = Context.get_parameter('USE_TENSORRT', default=False, direct=False)
        self.trt_weights = Context.get_file_path(trt_weights) if trt_weights else ''
        self.trt_plugin_library = Context.get_file_path(trt_plugin_library) if trt_plugin_library else ''
        self.non_trt_weights = Context.get_file_path(non_trt_weights) if non_trt_weights else ''
        self.device = device

        if use_tensorrt:
            jetpack_version = Context.get_parameter('JETPACK', direct=False)
            if jetpack_version == 6:
                LOGGER.info('Using TensorRT 10 (JetPack 6)')
                from .vehicle_tracking_with_tensorrt import VehicleTrackingTensorRT10
                self.model = VehicleTrackingTensorRT10(
                    weights=self.trt_weights,
                    plugin_library=self.trt_plugin_library,
                    device=self.device,
                )
            else:
                if jetpack_version not in [4, 5]:
                    LOGGER.warning(f'Unknown JETPACK version: {jetpack_version}, attempting to use TensorRT 8')
                else:
                    LOGGER.info(f'Using TensorRT 8 (JetPack {jetpack_version})')
                from .vehicle_tracking_with_tensorrt import VehicleTrackingTensorRT8
                self.model = VehicleTrackingTensorRT8(
                    weights=self.trt_weights,
                    plugin_library=self.trt_plugin_library,
                    device=self.device,
                )
        else:
            from .vehicle_tracking_without_tensorrt import VehicleTrackingWithoutTensorRT
            self.model = VehicleTrackingWithoutTensorRT(
                weights=self.non_trt_weights,
                device=self.device,
                high_score_threshold=high_score_threshold,
                low_score_threshold=low_score_threshold,
                new_track_threshold=new_track_threshold,
                match_score_threshold=match_score_threshold,
                secondary_iou_threshold=secondary_iou_threshold,
                max_age=max_age,
            )

        self.flops = getattr(self.model, 'flops', 0)

    def __call__(self, payload):
        return self.model(payload)
