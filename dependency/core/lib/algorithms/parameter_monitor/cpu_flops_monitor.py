import abc

import re
import subprocess

from core.lib.common import LOGGER
from core.lib.common import ClassFactory, ClassType
from .base_monitor import BaseMonitor

__all__ = ('CPUFlopsMonitor',)


@ClassFactory.register(ClassType.MON_PRAM, alias='cpu_flops')
class CPUFlopsMonitor(BaseMonitor, abc.ABC):
    def __init__(self, system):
        super().__init__(system)
        self.name = 'cpu_flops'

        self.device_cpu_flops = self.get_cpu_flops()

    def get_cpu_flops(self):
        try:
            info = self.parse_lscpu()
        except Exception as e:
            LOGGER.warning(f'Get cpu flops failed: {e}')
            LOGGER.exception(e)
            return 0

        if 'avx512f' in info['flags']:
            vector_width_bits = 512
            fma_units = 2
        elif 'avx2' in info['flags'] or 'avx' in info['flags']:
            vector_width_bits = 256
            fma_units = 2
        elif 'neon' in info['flags'] or 'asimd' in info['flags']:
            vector_width_bits = 128
            fma_units = 1
        else:
            vector_width_bits = 128
            fma_units = 1
        physical_cores = info['sockets'] * info['cores_per_socket']
        lanes = vector_width_bits // 32
        flops_per_cycle_per_core = lanes * 2 * fma_units
        freq_hz = info['max_mhz'] * 1e6
        peak_flops = physical_cores * flops_per_cycle_per_core * freq_hz

        peak_tflops = peak_flops / 1e12

        return peak_tflops

    @staticmethod
    def parse_lscpu():
        out = subprocess.check_output(['lscpu'], text=True)
        info = {
            'model_name': re.search(r'Model name:\s*(.+)', out).group(1).strip(),
            'sockets': int(re.search(r'Socket\(s\):\s*(\d+)', out).group(1)),
            'cores_per_socket': int(re.search(r'Core\(s\) per socket:\s*(\d+)', out).group(1)),
            'threads_per_core': int(re.search(r'Thread\(s\) per core:\s*(\d+)', out).group(1)),
            'max_mhz': float(re.search(r'CPU max MHz:\s*([\d.]+)', out).group(1)),
            'flags': re.search(r'Flags:\s*(.+)', out).group(1).split()
        }
        return info

    def get_parameter_value(self):
        return self.device_cpu_flops
