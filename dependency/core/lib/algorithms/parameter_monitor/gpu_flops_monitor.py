import abc
import re
import os
import platform
from typing import Tuple
import numpy as np

from .base_monitor import BaseMonitor

from core.lib.common import ClassFactory, ClassType, LOGGER

__all__ = ('GPUFlopsMonitor',)


@ClassFactory.register(ClassType.MON_PRAM, alias='gpu_flops')
class GPUFlopsMonitor(BaseMonitor, abc.ABC):
    def __init__(self, system):
        super().__init__(system)
        self.name = 'gpu_flops'

        self.DESKTOP_SM_COUNT_MAP = {
            "TITAN V": 80,  # Volta
            "TITAN RTX": 72,  # Turing
            "RTX 4090": 128,
            "RTX 4080": 76,
            "RTX 4070": 46,
            "RTX 3090": 82,
            "RTX 3080": 68,
            "RTX 3070": 46,
            "RTX 2080 Ti": 68,
            "RTX 2080": 46,
            "GTX 1080 Ti": 28,
            "A100": 108,  # Ampere
            "A40": 84,
            "A30": 56,
            "V100": 80,  # Volta
            "P100": 56,  # Pascal
            "T4": 40,  # Turing
            "QUADRO RTX 8000": 72,
            "QUADRO RTX 6000": 46,
            "RTX A6000": 84
        }

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
            'PASCAL': (128, 0, 0),
            'VOLTA': (64, 8, 64),
            'TURING': (64, 8, 64),
            'AMPERE': (128, 4, 256),
            'ADA_LOVELACE': (128, 4, 512)
        }

        # device specifications
        self.JETSON_DEVICE_SPECS = {
            'NVIDIA Jetson Nano': {
                'total_cores': 128,
                'max_freq_mhz': 921  # Maxwell GM20B
            },
            'NVIDIA Jetson TX2': {
                'total_cores': 256,
                'max_freq_mhz': 1300  # Pascal
            },
            'NVIDIA Jetson AGX Xavier': {
                'total_cores': 512,
                'max_freq_mhz': 1377  # Volta
            },
            'NVIDIA Jetson Xavier NX': {
                'total_cores': 384,
                'max_freq_mhz': 1100  # Volta
            },
            'NVIDIA Jetson AGX Orin': {
                'total_cores': 2048,
                'max_freq_mhz': 1300  # Ampere
            },
            'NVIDIA Jetson Orin Nano': {
                'total_cores': 1024,
                'max_freq_mhz': 1000  # Ampere Nano
            },
            'NVIDIA Jetson Orin NX': {
                'total_cores': 1024,
                'max_freq_mhz': 1000  # Ampere NX
            },
            'NVIDIA Jetson TX2 NX': {
                'total_cores': 256,
                'max_freq_mhz': 1300  # Pascal (TX2 NX)
            },
        }

        # device type alias
        self.JETSON_DEVICE_ALIASES = {
            r'jetson nano': 'NVIDIA Jetson Nano',
            r'jetson tx2': 'NVIDIA Jetson TX2',
            r'jetson agx xavier': 'NVIDIA Jetson AGX Xavier',
            r'jetson xavier nx': 'NVIDIA Jetson Xavier NX',
            r'jetson agx orin': 'NVIDIA Jetson AGX Orin',
            r'jetson orin nano': 'NVIDIA Jetson Orin Nano',
            r'jetson orin nx': 'NVIDIA Jetson Orin NX',
            r'jetson tx2 nx': 'NVIDIA Jetson TX2 NX',
        }

        # compute capability (cuda cores per SM)
        self.JETSON_CORES_PER_SM = {
            '5.3': 128,  # Nano (Maxwell)
            '6.2': 128,  # TX2, TX2 NX (Pascal)
            '7.2': 64,  # Xavier / Xavier NX (Volta)
            '8.6': 128,  # Orin Nano (Ampere)
            '8.7': 128,  # AGX Orin (Ampere)
        }

        # dual issue factor
        self.JETSON_DUAL_ISSUE_FACTOR = {
            '5.3': 1,  # Maxwell (Nano): single
            '6.2': 2,  # Pascal (TX2/TX2 NX): dual
            '7.2': 2,  # Volta (Xavier): dual
            '8.6': 2,  # Ampere (Orin Nano): dual
            '8.7': 2,  # Ampere (AGX Orin): dual
        }

    @staticmethod
    def load_nvml_library():
        import pynvml
        pynvml.nvmlInit()
        return pynvml

    def get_gpu_architecture_and_sm_count(self, device_name: str,
                                          compute_cap: Tuple[int, int]) -> Tuple[str, int]:
        """Determine the GPU architecture and SM count based on the device name and computing capability."""
        device_name_upper = device_name.upper()

        # Determines architecture based on computational capability.
        arch = self.DESKTOP_COMPUTE_CAPABILITY_MAP.get(compute_cap, 'UNKNOWN')

        # Try to obtain the SM count through the device name
        sm_count = 0
        for keyword, count in self.DESKTOP_SM_COUNT_MAP.items():
            if keyword in device_name_upper:
                sm_count = count
                break

        # Try to estimate based on device name
        if sm_count == 0:
            if compute_cap == (8, 9):  # Ada Lovelace
                sm_count = 128  # RTX 4090
            elif compute_cap == (8, 6):  # Ampere (RTX 30 series)
                sm_count = 82  # RTX 3090
            elif compute_cap == (7, 5):  # Turing (RTX 20 series)
                sm_count = 68  # RTX 2080 Ti
            elif compute_cap == (6, 2):  # Pascal (Jetson TX2)
                sm_count = 2

        return arch, sm_count

    @staticmethod
    def calculate_desktop_flops(
            sm_count: int,
            clock_speed: float,
            fp32_cores_per_sm: int):
        """Calculate desktop flops."""
        clock_hz = clock_speed * 1e6
        fp32_flops = sm_count * fp32_cores_per_sm * 2 * clock_hz

        return fp32_flops

    def normalize_module_name(self, module_str: str) -> str:
        """
        Standardized device name
        """
        normalized = module_str.lower().strip()
        normalized = re.sub(r'\([^)]*\)', '', normalized).strip()

        for suffix in ['module', 'board', 'developer kit', 'devkit']:
            normalized = normalized.replace(suffix, '').strip()

        for pattern, alias in self.JETSON_DEVICE_ALIASES.items():
            if re.search(pattern, normalized):
                return alias

        simple_name = normalized.replace(' ', '')
        for key in self.JETSON_DEVICE_SPECS:
            if simple_name == key.lower().replace(' ', ''):
                return key

        return module_str

    @staticmethod
    def calculate_jetson_flops(total_cores: int, cores_per_sm: int, freq_hz: float, dual_factor: int = 1):
        """
        Computational FLOPs for jetson devices
        """
        sm_count = total_cores // cores_per_sm if cores_per_sm else 0
        # FLOPs = SM count × per SM cores × frequency (Hz) × FMA  × dual factor
        return sm_count * cores_per_sm * freq_hz * 2 * dual_factor

    @staticmethod
    def get_gpu_frequency_info(gpu_data):
        """
        Extract frequency information from jtop data
        Return: (cur_freq_khz, max_freq_khz)
        """
        # Try the new jtop version of the data structure
        if hasattr(gpu_data, 'items'):
            try:
                # {'gp10b': {'freq': {'cur': 154000, 'max': 921000}, ...}}
                gpu_inner = next(iter(gpu_data.items()))[1]
                if isinstance(gpu_inner, dict):
                    freq_data = gpu_inner.get('freq', {})
                    return freq_data.get('cur', 0), freq_data.get('max', 0)
            except Exception:
                pass

        # Try the old jtop version of the data structure
        freq_dict = getattr(gpu_data, 'freq', None)
        if freq_dict is None and hasattr(gpu_data, 'get'):
            freq_dict = gpu_data.get('freq', {})

        if isinstance(freq_dict, dict):
            return freq_dict.get('cur', 0), freq_dict.get('max', 0)

        return 0, 0

    @staticmethod
    def is_jetson_device() -> bool:
        """Check if the current device is a Jetson device"""
        try:
            with open('/proc/device-tree/model', 'r') as f:
                model = f.read().lower()
                if 'jetson' in model:
                    return True
        except:
            pass

        if os.path.exists('/etc/nv_tegra_release'):
            return True

        if platform.machine().lower() in ['aarch64', 'arm64']:
            if os.path.exists('/sys/module/tegra_fuse/parameters/tegra_chip_id'):
                return True

        return False

    def get_desktop_device_gpu_flops(self):
        pynvml = self.load_nvml_library()

        device_count = pynvml.nvmlDeviceGetCount()

        flops_results = []

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)

            name_value = pynvml.nvmlDeviceGetName(handle)
            device_name = name_value.decode('utf-8', errors='replace')

            # Get computing capability
            cc_major = pynvml.nvmlDeviceGetCudaComputeCapability(handle)[0]
            cc_minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)[1]
            compute_cap = (cc_major, cc_minor)

            # Get architecture and sm count
            arch, sm_count = self.get_gpu_architecture_and_sm_count(device_name, compute_cap)

            # Get clock frequency
            try:
                max_clock = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_SM)
            except (AttributeError, pynvml.NVMLError):
                try:
                    max_clock = pynvml.nvmlDeviceGetClock(handle, pynvml.NVML_CLOCK_SM,
                                                          pynvml.NVML_CLOCK_ID_CURRENT)
                except:
                    max_clock = 0

            # Get architecture parameters
            if arch in self.DESKTOP_ARCH_PARAMS:
                fp32_cores, _, _ = self.DESKTOP_ARCH_PARAMS[arch]
            else:
                raise Exception(f"unknown architecture for {device_name}.")

            # Calculate FLOPs
            flops = self.calculate_desktop_flops(sm_count, max_clock, fp32_cores)

            flops_results.append(flops / 1e12)

        pynvml.nvmlShutdown()

        return np.mean(flops_results)

    def get_jetson_device_gpu_flops(self):
        from jtop import jtop
        with jtop() as jetson:
            if not jetson.ok():
                raise Exception("jtop initialization failed for jetson device")

            # Get static hardware information
            hw = jetson.board.get('hardware', {})
            raw_module = hw.get('Module', 'Unknown Jetson')
            cc = hw.get('CUDA Arch BIN', '')

            # Normalize device module name
            normalized_module = self.normalize_module_name(raw_module)

            # Read gpu frequency
            _, max_freq_khz = self.get_gpu_frequency_info(jetson.gpu)

            # Query for device specification
            device_spec = self.JETSON_DEVICE_SPECS.get(normalized_module, {})
            total_cores = device_spec.get('total_cores')
            default_max_freq_mhz = device_spec.get('max_freq_mhz', 0)

            cores_per_sm = self.JETSON_CORES_PER_SM.get(cc)
            dual_factor = self.JETSON_DUAL_ISSUE_FACTOR.get(cc, 1)

            # Calculate FLOPs
            if total_cores is None or cores_per_sm is None:
                raise Exception(f"unsupported device or computing capability: "
                                f"{cc} is not supported for {normalized_module}")

            max_freq_hz = (max_freq_khz or default_max_freq_mhz * 1000) * 1e3
            theo_peak = self.calculate_jetson_flops(total_cores, cores_per_sm, max_freq_hz, dual_factor)

            return theo_peak / 1e12

    def get_gpu_flops(self):
        try:
            if self.is_jetson_device():
                return self.get_jetson_device_gpu_flops()
            else:
                return self.get_desktop_device_gpu_flops()
        except Exception as e:
            LOGGER.warning(f'Get gpu flops failed: {e}')
            LOGGER.exception(e)
            return 0

    def get_parameter_value(self):
        fp32_flops = self.get_gpu_flops()
        return fp32_flops
