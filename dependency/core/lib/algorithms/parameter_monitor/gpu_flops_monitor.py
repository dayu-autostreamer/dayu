import abc
import os
import platform

from .base_monitor import BaseMonitor

from core.lib.common import ClassFactory, ClassType, LOGGER

__all__ = ('GPUFlopsMonitor',)


@ClassFactory.register(ClassType.MON_PRAM, alias='gpu_flops')
class GPUFlopsMonitor(BaseMonitor, abc.ABC):
    def __init__(self, system):
        super().__init__(system)
        self.name = 'gpu_flops'

        """        
        # Basic GPU Information
        self.DESKTOP_COMPUTE_CAPABILITY_MAP = {
            (6, 0): 'PASCAL',  # GP100
            (6, 1): 'PASCAL',  # GP102/104/106
            (7, 0): 'VOLTA',  # GV100
            (7, 5): 'TURING',  # TU102/104/106
            (8, 0): 'AMPERE',  # GA100
            (8, 6): 'AMPERE',  # GA102/104 (RTX 30)
            (8, 9): 'ADA_LOVELACE'  # AD102/104 (RTX 40)
        }

        self.DESKTOP_ARCH_PARAMS = {
            'PASCAL': (128, 2),
            'VOLTA': (64, 2),
            'TURING': (64, 2),
            'AMPERE': (128, 2),
            'ADA_LOVELACE': (128, 2),
        }
        """

        self.DESKTOP_CORES_PER_SM = {
            (6, 0): 128,  # GP100
            (6, 1): 128,  # GP102/104/106
            (7, 0): 64,  # GV100
            (7, 5): 64,  # TU102/104/106
            (8, 0): 128,  # GA100
            (8, 6): 128,  # GA102/104 (RTX 30)
            (8, 9): 128,  # AD102/104 (RTX 40)
        }

        self.DESKTOP_DUAL_ISSUE = {
            (6, 0): 2,  # GP100
            (6, 1): 2,  # GP102/104/106
            (7, 0): 2,  # GV100
            (7, 5): 2,  # TU102/104/106
            (8, 0): 2,  # GA100
            (8, 6): 2,  # GA102/104 (RTX 30)
            (8, 9): 2,  # AD102/104 (RTX 40)
        }

        self.JETSON_CORES_PER_SM = {
            (5, 3): 128,  # Jetson Nano (Maxwell)
            (6, 2): 128,  # TX2 / TX2 NX (Pascal)
            (7, 2): 64,  # Xavier / Xavier NX (Volta)
            (8, 6): 128,  # Orin Nano (Ampere)
            (8, 7): 128,  # AGX Orin / Orin NX (Ampere)
        }
        self.JETSON_DUAL_ISSUE = {
            (5, 3): 1,  # Maxwell (Nano): single
            (6, 2): 2,  # Pascal (TX2/TX2 NX): dual
            (7, 2): 2,  # Volta (Xavier): dual
            (8, 6): 2,  # Ampere (Orin Nano): dual
            (8, 7): 2,  # Ampere (AGX Orin): dual
        }

    @staticmethod
    def load_pycuda():
        import pycuda.driver as cuda
        cuda.init()
        return cuda

    @staticmethod
    def calculate_flops(sm_count: int, fp32_cores_per_sm: int, clock_freq_hz: float, dual_factor) -> float:
        return sm_count * fp32_cores_per_sm * clock_freq_hz * 2 * dual_factor

    @staticmethod
    def is_jetson_device() -> bool:
        """Check if the current device is a Jetson device"""
        if os.path.exists('/etc/nv_tegra_release'):
            return True

        if platform.machine().lower() in ['aarch64', 'arm64']:
            if os.path.exists('/sys/module/tegra_fuse/parameters/tegra_chip_id'):
                return True

        return False

    def get_device_fp32_flops(self, is_jetson: bool = False):
        cuda = self.load_pycuda()
        total_flops = 0
        CORES_PER_SM = self.JETSON_CORES_PER_SM if is_jetson else self.DESKTOP_CORES_PER_SM
        DEVICE_DUAL_ISSUE = self.JETSON_DUAL_ISSUE if is_jetson else self.DESKTOP_DUAL_ISSUE
        for idx in range(cuda.Device.count()):
            device = cuda.Device(idx)
            device_name = device.name().lower()
            max_freq_hz = device.get_attribute(cuda.device_attribute.CLOCK_RATE)
            capability = device.compute_capability()
            mp_count = device.get_attribute(cuda.device_attribute.MULTIPROCESSOR_COUNT)
            fp32_cores_per_sm = CORES_PER_SM.get(capability)
            dual_factor = DEVICE_DUAL_ISSUE.get(capability)
            if fp32_cores_per_sm is None or dual_factor is None:
                raise Exception(f"Unsupported device or computing capability: {capability} for {device_name}")
            total_flops += self.calculate_flops(mp_count, fp32_cores_per_sm, max_freq_hz, dual_factor)

        return total_flops / cuda.Device.count() / 1e12

    def get_gpu_flops(self):
        try:
            return self.get_device_fp32_flops(self.is_jetson_device())
        except Exception as e:
            LOGGER.warning(f'Get gpu flops failed: {e}')
            LOGGER.exception(e)
            return 0

    def get_parameter_value(self):
        fp32_flops = self.get_gpu_flops()
        return fp32_flops
