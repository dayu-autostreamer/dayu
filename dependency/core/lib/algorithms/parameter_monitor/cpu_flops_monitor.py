import abc
from typing import Tuple
import platform
import os
import re
import subprocess

from core.lib.common import LOGGER

from .base_monitor import BaseMonitor

from core.lib.common import ClassFactory, ClassType

__all__ = ('CPUFlopsMonitor',)


@ClassFactory.register(ClassType.MON_PRAM, alias='cpu_flops')
class CPUFlopsMonitor(BaseMonitor, abc.ABC):
    def __init__(self, system):
        super().__init__(system)
        self.name = 'cpu_flops'

        self.cores, self.freq, self.arch, self.instruction_set = self.get_cpu_info()

    @staticmethod
    def get_cpu_info() -> Tuple[int, float, str, str]:
        """Get the number of CPU cores, maximum frequency, architecture, and instruction set."""
        # Get the number of physical cores
        if platform.system() == "Linux":
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    cpuinfo = f.read()
                cores = int(re.search(r'cpu cores\s*:\s*(\d+)', cpuinfo).group(1))
            except Exception as e:
                LOGGER.warning(f'Obtaining CPU cores failed: {e}')
                cores = os.cpu_count() // 2 or os.cpu_count() or 1
        else:
            cores = os.cpu_count() or 1

        # Get CPU frequency and architecture
        freq, arch = None, platform.machine()
        try:
            if platform.system() == "Linux":
                with open('/proc/cpuinfo', 'r') as f:
                    for line in f:
                        if 'cpu MHz' in line:
                            freq = max(freq or 0, float(line.split(':')[1].strip()))
                    if freq: freq /= 1000  # transform to GHz
            elif platform.system() == "Windows":
                # Use wmic to get frequency
                output = subprocess.check_output(
                    'wmic cpu get maxclockspeed',
                    shell=True, text=True, stderr=subprocess.STDOUT
                )
                freq = max(float(x) for x in re.findall(r'\d+', output)) / 1000
        except Exception as e:
            LOGGER.warning(f'Obtaining CPU frequency and architecture failed: {e}')

        # Backup frequency acquisition
        if not freq:
            try:
                freq = float(subprocess.check_output(
                    ["lscpu", "-b", "-p=MAXMHZ"],
                    text=True, stderr=subprocess.DEVNULL
                ).split('\n')[1].strip().split(',')[0]) / 1000
            except Exception as e:
                LOGGER.warning(f'Obtaining CPU frequency and architecture failed: {e}')
                freq = 2.0  # default

        # Detect the instruction set
        instruction_set = ""
        try:
            lscpu = subprocess.check_output("lscpu", text=True, stderr=subprocess.DEVNULL)
            if 'avx512' in lscpu.lower():
                instruction_set = "AVX512"
            elif 'avx2' in lscpu.lower():
                instruction_set = "AVX2"
            elif 'avx' in lscpu.lower():
                instruction_set = "AVX"
            elif 'neon' in lscpu.lower() or 'asimd' in lscpu.lower():
                instruction_set = "NEON"
        except Exception as e:
            LOGGER.warning(f'Obtaining CPU instruction set failed: {e}')
            # ARM device check
            if 'arm' in arch.lower() or 'aarch' in arch.lower():
                instruction_set = "NEON"
            else:
                instruction_set = "SSE"

        return cores, freq, arch, instruction_set

    def calculate_flops(self) -> Tuple[float, float]:
        """
        Calculate the theoretical peak FLOPS for CPU
        SP=FLOPS32, DP=FLOPS64
        Unit: GFLOPS
        """

        # Process x86 architecture
        if 'x86' in self.arch or 'amd' in self.arch.lower():
            if self.instruction_set == "AVX512":
                ops_per_cycle_sp = 64  # 512 bit / 32 bit * 2 (FMA)
                ops_per_cycle_dp = 32  # 512 bit / 64 bit * 2 (FMA)
            elif self.instruction_set == "AVX2":
                ops_per_cycle_sp = 32  # 256 bit / 32 bit * 2 (FMA)
                ops_per_cycle_dp = 16  # 256 bit / 64 bit * 2 (FMA)
            elif self.instruction_set == "AVX":
                ops_per_cycle_sp = 16  # 256 bit / 32 bit * 1 (no FMA)
                ops_per_cycle_dp = 8  # 256 bit / 64 bit * 1 (no FMA)
            else:  # SSE
                ops_per_cycle_sp = 8  # 128 bit / 32 bit * 1
                ops_per_cycle_dp = 4  # 128 bit / 64 bit * 1

        # Process arm architecture
        elif 'arm' in self.arch.lower() or 'aarch' in self.arch.lower():
            if self.instruction_set == "NEON":
                ops_per_cycle_sp = 8  # 128 bit / 32 bit * 2(FMA)
                ops_per_cycle_dp = 4  # 128 bit / 64 bit * 2(FMA)
            else:
                ops_per_cycle_sp = 2
                ops_per_cycle_dp = 1

        # Other architectures use conservative estimates
        else:
            ops_per_cycle_sp = 2
            ops_per_cycle_dp = 1

        # Calculate the theoretical peak value (FLOPS = Cores * GHz * Operations / cycles) GFLOPS
        peak_sp = self.cores * self.freq * ops_per_cycle_sp
        peak_dp = self.cores * self.freq * ops_per_cycle_dp
        return peak_sp, peak_dp

    def get_parameter_value(self):
        cpu_flops, _ = self.calculate_flops()
        return cpu_flops
