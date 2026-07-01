import abc
import glob
import os
import platform
import re
import shutil
import subprocess

from core.lib.common import ClassFactory, ClassType, LOGGER
from .base_monitor import BaseMonitor

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

        self._is_jetson = self.is_jetson_device()
        self._device_meta = None
        self._device_meta_loader_id = None
        self._device_meta_is_jetson = None

    @staticmethod
    def load_pycuda():
        import pycuda.driver as cuda
        cuda.init()
        return cuda

    @staticmethod
    def calculate_flops(sm_count: int, fp32_cores_per_sm: int, clock_freq_khz: float, dual_factor) -> float:
        # PyCUDA reports CLOCK_RATE in kHz. Convert to Hz so the public monitor
        # value below is GFLOP/s, which matches the giga-scale model FLOPs.
        clock_freq_hz = float(clock_freq_khz) * 1000.0
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

    def _get_device_meta(self, is_jetson: bool = False):
        loader = self.load_pycuda
        loader_id = id(loader)
        if self._device_meta is not None:
            if self._device_meta_loader_id is None:
                return self._device_meta
            if (
                self._device_meta_loader_id == loader_id
                and self._device_meta_is_jetson == bool(is_jetson)
            ):
                return self._device_meta
        cuda = loader()
        CORES_PER_SM = self.JETSON_CORES_PER_SM if is_jetson else self.DESKTOP_CORES_PER_SM
        DEVICE_DUAL_ISSUE = self.JETSON_DUAL_ISSUE if is_jetson else self.DESKTOP_DUAL_ISSUE
        meta = []
        for idx in range(cuda.Device.count()):
            device = cuda.Device(idx)
            device_name = device.name().lower()
            max_freq_khz = device.get_attribute(cuda.device_attribute.CLOCK_RATE)
            capability = device.compute_capability()
            mp_count = device.get_attribute(cuda.device_attribute.MULTIPROCESSOR_COUNT)
            fp32_cores_per_sm = CORES_PER_SM.get(capability)
            dual_factor = DEVICE_DUAL_ISSUE.get(capability)
            if fp32_cores_per_sm is None or dual_factor is None:
                raise Exception(f"Unsupported device or computing capability: {capability} for {device_name}")
            meta.append({
                "idx": idx,
                "name": device_name,
                "max_freq_khz": float(max_freq_khz),
                "capability": capability,
                "sm_count": int(mp_count),
                "fp32_cores_per_sm": int(fp32_cores_per_sm),
                "dual_factor": float(dual_factor),
            })
        self._device_meta = meta
        self._device_meta_loader_id = loader_id
        self._device_meta_is_jetson = bool(is_jetson)
        return meta

    def get_device_fp32_flops(self, is_jetson: bool = False):
        meta = self._get_device_meta(is_jetson)
        if not meta:
            return 0

        current_clocks = self._read_current_clock_rates_khz(len(meta), is_jetson)
        total_flops = 0
        for idx, device_meta in enumerate(meta):
            clock_freq_khz = device_meta["max_freq_khz"]
            if idx < len(current_clocks) and current_clocks[idx] is not None and current_clocks[idx] > 0:
                clock_freq_khz = min(float(current_clocks[idx]), device_meta["max_freq_khz"])
            total_flops += self.calculate_flops(
                device_meta["sm_count"],
                device_meta["fp32_cores_per_sm"],
                clock_freq_khz,
                device_meta["dual_factor"],
            )

        return total_flops / len(meta) / 1e9

    def _read_current_clock_rates_khz(self, device_count: int, is_jetson: bool):
        """Read current GPU clock rates in kHz.

        PyCUDA's CLOCK_RATE is a static maximum clock. Hedger uses gpu_flops as
        the current device capacity, so frequency-capped Jetsons need the live
        devfreq/NVML/nvidia-smi clock when available. The method returns one
        value per CUDA device and falls back to an empty list when no runtime
        clock backend is available.
        """
        readers = (
            [
                self._get_current_clocks_via_jetson_sysfs,
                self._get_current_clocks_via_nvml,
                self._get_current_clocks_via_tegrastats,
                self._get_current_clocks_via_nvidia_smi,
            ]
            if is_jetson else
            [
                self._get_current_clocks_via_nvml,
                self._get_current_clocks_via_nvidia_smi,
            ]
        )
        for reader in readers:
            try:
                clocks = reader()
            except Exception:
                clocks = None
            clocks = self._normalize_clock_list(clocks, device_count)
            if clocks:
                return clocks
        return []

    @staticmethod
    def _normalize_clock_list(clocks, device_count: int):
        if not clocks or device_count <= 0:
            return []
        normalized = []
        for value in clocks:
            khz = GPUFlopsMonitor._clock_value_to_khz(value)
            if khz is not None and khz > 0:
                normalized.append(khz)
        if not normalized:
            return []
        if len(normalized) == 1 and device_count > 1:
            normalized = normalized * device_count
        if len(normalized) < device_count:
            normalized.extend([None] * (device_count - len(normalized)))
        return normalized[:device_count]

    @staticmethod
    def _clock_value_to_khz(value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        if value <= 0:
            return None
        # sysfs devfreq and BPMP debug files often use Hz; PyCUDA uses kHz;
        # NVML/nvidia-smi/tegrastats report MHz. Normalize all to kHz.
        if value >= 10_000_000:
            return value / 1000.0
        if value >= 10_000:
            return value
        return value * 1000.0

    @staticmethod
    def _get_current_clocks_via_nvml():
        try:
            import pynvml
        except Exception:
            return None
        try:
            pynvml.nvmlInit()
            try:
                count = pynvml.nvmlDeviceGetCount()
                clocks = []
                for idx in range(count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                    clock_mhz = None
                    for clock_type_name in ("NVML_CLOCK_SM", "NVML_CLOCK_GRAPHICS"):
                        clock_type = getattr(pynvml, clock_type_name, None)
                        if clock_type is None:
                            continue
                        try:
                            clock_mhz = pynvml.nvmlDeviceGetClockInfo(handle, clock_type)
                            break
                        except Exception:
                            continue
                    if clock_mhz is not None:
                        clocks.append(clock_mhz)
                return clocks or None
            finally:
                try:
                    pynvml.nvmlShutdown()
                except Exception:
                    pass
        except Exception:
            return None

    @staticmethod
    def _get_current_clocks_via_nvidia_smi(timeout_sec: float = 1.0):
        if shutil.which("nvidia-smi") is None:
            return None
        queries = ("clocks.current.sm", "clocks.current.graphics")
        for query in queries:
            try:
                result = subprocess.run(
                    ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=timeout_sec,
                    check=False,
                    text=True,
                )
                if result.returncode != 0:
                    continue
                clocks = []
                for line in result.stdout.strip().splitlines():
                    match = re.search(r"[-+]?\d+(?:\.\d+)?", line)
                    if match:
                        clocks.append(float(match.group(0)))
                if clocks:
                    return clocks
            except Exception:
                continue
        return None

    @staticmethod
    def _get_current_clocks_via_jetson_sysfs():
        patterns = [
            "/sys/devices/gpu.0/devfreq/*/cur_freq",
            "/sys/devices/*gpu*/devfreq/*/cur_freq",
            "/sys/devices/*gv11b*/devfreq/*/cur_freq",
            "/sys/devices/*nvgpu*/devfreq/*/cur_freq",
            "/sys/devices/platform/*gpu*/devfreq/*/cur_freq",
            "/sys/devices/platform/host1x/*gpu*/devfreq/*/cur_freq",
            "/sys/class/devfreq/*gpu*/cur_freq",
            "/sys/class/devfreq/*gv11b*/cur_freq",
            "/sys/class/devfreq/*nvgpu*/cur_freq",
            "/sys/kernel/debug/bpmp/debug/clk/gpu/rate",
            "/sys/kernel/debug/bpmp/debug/clk/gpusys/rate",
        ]
        paths = []
        for pattern in patterns:
            paths.extend(glob.glob(pattern))
        unique_paths = []
        seen = set()
        for path in paths:
            if path not in seen and os.path.isfile(path):
                seen.add(path)
                unique_paths.append(path)
        clocks = []
        for path in unique_paths:
            try:
                with open(path, "r", encoding="utf-8") as fp:
                    raw = fp.read().strip()
                match = re.search(r"[-+]?\d+(?:\.\d+)?", raw)
                if match:
                    clocks.append(float(match.group(0)))
            except Exception:
                continue
        if not clocks:
            return None
        # Jetson boards normally expose one integrated GPU through several sysfs
        # aliases. Use the largest live clock if duplicate paths are present.
        return [max(clocks)]

    @staticmethod
    def _get_current_clocks_via_tegrastats():
        if shutil.which("tegrastats") is None:
            return None
        try:
            proc = subprocess.Popen(["tegrastats"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                line = proc.stdout.readline().strip() if proc.stdout else ""
            finally:
                try:
                    proc.terminate()
                except Exception:
                    pass
            if not line:
                return None
            match = re.search(r"GR3D_FREQ\s+\d+%(?:@(?:\[([^\]]+)\]|(\d+(?:\.\d+)?)))?", line)
            if not match:
                return None
            if match.group(1):
                values = []
                for part in match.group(1).split(","):
                    part = part.strip()
                    if part:
                        values.append(float(part))
                return values or None
            if match.group(2):
                return [float(match.group(2))]
            return None
        except Exception:
            return None

    def get_gpu_flops(self):
        try:
            return self.get_device_fp32_flops(self._is_jetson)
        except Exception as e:
            LOGGER.warning(f'Get gpu flops failed: {e}')
            LOGGER.exception(e)
            return 0

    def get_parameter_value(self):
        return self.get_gpu_flops()
