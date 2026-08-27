import abc
import glob
import os
import platform
import re
import shutil
import subprocess
import time

from core.lib.common import ClassFactory, ClassType, LOGGER
from .base_monitor import BaseMonitor

__all__ = ('GPUFlopsMonitor',)


@ClassFactory.register(ClassType.MON_PRAM, alias='gpu_flops')
class GPUFlopsMonitor(BaseMonitor, abc.ABC):
    """Report the current single-precision GPU throughput in GFLOP/s."""

    MIN_LAST_KNOWN_GOOD_SECONDS = 15.0

    DESKTOP_FP32_CORES_PER_SM = {
        (6, 0): 64,   # GP100
        (6, 1): 128,  # GP102/104/106
        (7, 0): 64,   # GV100
        (7, 5): 64,   # TU102/104/106
        (8, 0): 64,   # GA100
        (8, 6): 128,  # GA102/104 (RTX 30)
        (8, 9): 128,  # AD102/104 (RTX 40)
    }

    # soc_id -> immutable GPU architecture. Power-mode configuration determines
    # the enabled TPCs while devfreq changes continuously, so sample both from
    # the same GPU sysfs tree instead of assuming a product-level peak.
    JETSON_SOC_PROFILES = {
        0x21: {
            "name": "gm20b",
            "capability": (5, 3),
            "sms_per_tpc": 1,
            "fp32_cores_per_sm": 128,
            "max_tpc_count": 1,
        },
        0x18: {
            "name": "gp10b",
            "capability": (6, 2),
            "sms_per_tpc": 1,
            "fp32_cores_per_sm": 128,
            "max_tpc_count": 2,
        },
        0x19: {
            "name": "gv11b",
            "capability": (7, 2),
            "sms_per_tpc": 2,
            "fp32_cores_per_sm": 64,
            "max_tpc_count": 4,
        },
        0x23: {
            "name": "ga10b",
            "capability": (8, 7),
            "sms_per_tpc": 2,
            "fp32_cores_per_sm": 128,
            "max_tpc_count": 8,
        },
    }
    JETSON_SOC_ID_PATHS = (
        "/sys/devices/soc0/soc_id",
        "/sys/module/tegra_fuse/parameters/tegra_chip_id",
    )
    JETSON_DEVFREQ_PATTERNS = (
        "/sys/class/devfreq/*gpu*",
        "/sys/class/devfreq/*gv11b*",
        "/sys/class/devfreq/*nvgpu*",
        "/sys/devices/gpu.0/devfreq/*",
        "/sys/devices/*gpu*/devfreq/*",
        "/sys/devices/*gv11b*/devfreq/*",
        "/sys/devices/*nvgpu*/devfreq/*",
        "/sys/devices/platform/*gpu*/devfreq/*",
        "/sys/devices/platform/*/*gpu*/devfreq/*",
        "/sys/devices/platform/host1x/*gpu*/devfreq/*",
    )

    def __init__(self, system):
        super().__init__(system)
        self.name = 'gpu_flops'
        self._is_jetson = self.is_jetson_device()
        self._device_meta = None
        self._jetson_profile = None
        self._jetson_paths = None
        self._last_error = None
        self._last_success_at = None
        monitor_interval = float(getattr(system, 'monitor_interval', 5.0) or 5.0)
        self._last_known_good_seconds = max(
            self.MIN_LAST_KNOWN_GOOD_SECONDS,
            3.0 * monitor_interval,
        )

    @staticmethod
    def load_pycuda():
        import pycuda.driver as cuda
        cuda.init()
        return cuda

    @staticmethod
    def calculate_flops(sm_count: int, fp32_cores_per_sm: int, clock_freq_khz: float) -> float:
        """Calculate FP32 operations per second; one FMA is two operations."""
        clock_freq_hz = float(clock_freq_khz) * 1000.0
        return sm_count * fp32_cores_per_sm * clock_freq_hz * 2

    @staticmethod
    def is_jetson_device() -> bool:
        if os.path.exists('/etc/nv_tegra_release'):
            return True
        return (
            platform.machine().lower() in {'aarch64', 'arm64'}
            and any(os.path.exists(path) for path in GPUFlopsMonitor.JETSON_SOC_ID_PATHS)
        )

    @staticmethod
    def _read_text(path: str) -> str:
        with open(path, 'r', encoding='utf-8') as fp:
            return fp.read().strip()

    @classmethod
    def _read_jetson_soc_id(cls) -> int:
        for path in cls.JETSON_SOC_ID_PATHS:
            try:
                raw = cls._read_text(path)
            except OSError:
                continue
            try:
                return int(raw, 16 if raw.lower().startswith('0x') else 10)
            except ValueError:
                continue
        raise RuntimeError('Jetson soc_id is unavailable')

    @classmethod
    def _parse_tpc_mask(cls, raw: str) -> int:
        raw = raw.strip()
        try:
            mask = int(raw, 0)
        except ValueError:
            mask = int(raw, 16)
        if mask <= 0:
            raise RuntimeError(f'Jetson tpc_fs_mask is not positive: {raw!r}')
        return mask

    @staticmethod
    def _read_positive_int(path: str, field: str) -> int:
        raw = GPUFlopsMonitor._read_text(path)
        try:
            value = int(raw, 10)
        except ValueError as exc:
            raise RuntimeError(f'Jetson {field} is invalid: {raw!r}') from exc
        if value <= 0:
            raise RuntimeError(f'Jetson {field} is not positive: {value}')
        return value

    @classmethod
    def _discover_jetson_paths(cls):
        candidates = []
        seen = set()
        for pattern in cls.JETSON_DEVFREQ_PATTERNS:
            for path in glob.glob(pattern):
                devfreq_dir = os.path.realpath(path)
                if devfreq_dir in seen or not os.path.isdir(devfreq_dir):
                    continue
                seen.add(devfreq_dir)
                candidates.append(devfreq_dir)

        for devfreq_dir in sorted(candidates):
            cur_freq = os.path.join(devfreq_dir, 'cur_freq')
            max_freq = os.path.join(devfreq_dir, 'max_freq')
            if not os.path.isfile(cur_freq) or not os.path.isfile(max_freq):
                continue
            parent = os.path.dirname(devfreq_dir)
            gpu_root = os.path.dirname(parent) if os.path.basename(parent) == 'devfreq' else parent
            tpc_candidates = (
                os.path.join(gpu_root, 'tpc_fs_mask'),
                os.path.join(os.path.realpath(os.path.join(devfreq_dir, 'device')), 'tpc_fs_mask'),
            )
            tpc_mask = next((path for path in tpc_candidates if os.path.isfile(path)), None)
            if tpc_mask:
                return {
                    'tpc_mask': tpc_mask,
                    'cur_freq': cur_freq,
                    'max_freq': max_freq,
                }
        raise RuntimeError('Jetson GPU devfreq/tpc_fs_mask sysfs files are unavailable')

    def _get_jetson_profile(self):
        if self._jetson_profile is not None:
            return self._jetson_profile
        soc_id = self._read_jetson_soc_id()
        profile = self.JETSON_SOC_PROFILES.get(soc_id)
        if profile is None:
            raise RuntimeError(f'unsupported Jetson soc_id: {soc_id} (0x{soc_id:x})')
        self._jetson_profile = profile
        return profile

    def _get_jetson_paths(self):
        if self._jetson_paths is None:
            self._jetson_paths = self._discover_jetson_paths()
        return self._jetson_paths

    def _read_jetson_state(self, profile):
        paths = self._get_jetson_paths()
        try:
            for _ in range(2):
                mask_before = self._parse_tpc_mask(self._read_text(paths['tpc_mask']))
                max_before = self._read_positive_int(paths['max_freq'], 'max_freq')
                current_hz = self._read_positive_int(paths['cur_freq'], 'cur_freq')
                max_after = self._read_positive_int(paths['max_freq'], 'max_freq')
                mask_after = self._parse_tpc_mask(self._read_text(paths['tpc_mask']))
                if mask_before != mask_after or max_before != max_after:
                    continue
                # Monitor images for JetPack 4 still use Python 3.8, which does
                # not provide int.bit_count().
                active_tpc_count = bin(mask_after).count('1')
                if active_tpc_count > profile['max_tpc_count']:
                    raise RuntimeError(
                        f"Jetson active TPC count {active_tpc_count} exceeds "
                        f"{profile['name']} limit {profile['max_tpc_count']}"
                    )
                if current_hz > max_after:
                    raise RuntimeError(
                        f'Jetson cur_freq {current_hz} exceeds max_freq {max_after}'
                    )
                return active_tpc_count, current_hz
        except OSError:
            self._jetson_paths = None
            raise
        raise RuntimeError('Jetson GPU topology or frequency changed during sampling')

    def _get_jetson_fp32_flops(self) -> float:
        profile = self._get_jetson_profile()
        active_tpc_count, current_hz = self._read_jetson_state(profile)
        sm_count = active_tpc_count * profile['sms_per_tpc']
        flops = self.calculate_flops(
            sm_count,
            profile['fp32_cores_per_sm'],
            current_hz / 1000.0,
        )
        return flops / 1e9

    def _get_device_meta(self):
        if self._device_meta is not None:
            return self._device_meta
        cuda = self.load_pycuda()
        meta = []
        for idx in range(cuda.Device.count()):
            device = cuda.Device(idx)
            device_name = device.name().lower()
            capability = device.compute_capability()
            fp32_cores_per_sm = self.DESKTOP_FP32_CORES_PER_SM.get(capability)
            if fp32_cores_per_sm is None:
                raise RuntimeError(
                    f'unsupported computing capability {capability} for {device_name}'
                )
            meta.append({
                'idx': idx,
                'name': device_name,
                'max_freq_khz': float(device.get_attribute(cuda.device_attribute.CLOCK_RATE)),
                'capability': capability,
                'sm_count': int(device.get_attribute(cuda.device_attribute.MULTIPROCESSOR_COUNT)),
                'fp32_cores_per_sm': fp32_cores_per_sm,
            })
        self._device_meta = meta
        return meta

    def get_device_fp32_flops(self, is_jetson: bool = False):
        if is_jetson:
            return self._get_jetson_fp32_flops()

        meta = self._get_device_meta()
        if not meta:
            return 0.0
        current_clocks = self._read_current_clock_rates_khz(len(meta))
        total_flops = 0.0
        for idx, device_meta in enumerate(meta):
            clock_freq_khz = device_meta['max_freq_khz']
            if idx < len(current_clocks) and current_clocks[idx] is not None:
                clock_freq_khz = min(float(current_clocks[idx]), device_meta['max_freq_khz'])
            total_flops += self.calculate_flops(
                device_meta['sm_count'],
                device_meta['fp32_cores_per_sm'],
                clock_freq_khz,
            )
        return total_flops / len(meta) / 1e9

    def _read_current_clock_rates_khz(self, device_count: int):
        for reader in (self._get_current_clocks_via_nvml, self._get_current_clocks_via_nvidia_smi):
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
            normalized *= device_count
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
        if value >= 10_000_000:  # Hz from sysfs
            return value / 1000.0
        if value >= 10_000:  # kHz from CUDA
            return value
        return value * 1000.0  # MHz from NVML/nvidia-smi

    @staticmethod
    def _get_current_clocks_via_nvml():
        try:
            import pynvml
        except Exception:
            return None
        try:
            pynvml.nvmlInit()
            try:
                clocks = []
                for idx in range(pynvml.nvmlDeviceGetCount()):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                    for clock_type_name in ('NVML_CLOCK_SM', 'NVML_CLOCK_GRAPHICS'):
                        clock_type = getattr(pynvml, clock_type_name, None)
                        if clock_type is None:
                            continue
                        try:
                            clocks.append(pynvml.nvmlDeviceGetClockInfo(handle, clock_type))
                            break
                        except Exception:
                            continue
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
        if shutil.which('nvidia-smi') is None:
            return None
        for query in ('clocks.current.sm', 'clocks.current.graphics'):
            try:
                result = subprocess.run(
                    ['nvidia-smi', f'--query-gpu={query}', '--format=csv,noheader,nounits'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=timeout_sec,
                    check=False,
                    text=True,
                )
            except Exception:
                continue
            if result.returncode != 0:
                continue
            clocks = []
            for line in result.stdout.strip().splitlines():
                match = re.search(r'[-+]?\d+(?:\.\d+)?', line)
                if match:
                    clocks.append(float(match.group(0)))
            if clocks:
                return clocks
        return None

    def get_gpu_flops(self):
        try:
            value = self.get_device_fp32_flops(self._is_jetson)
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
            if error != self._last_error:
                LOGGER.warning(f'GPU FLOPS monitor unavailable: {error}')
            self._last_error = error
            return None
        if self._last_error is not None:
            LOGGER.info('GPU FLOPS monitor recovered')
            self._last_error = None
        return value

    def get_parameter_value(self):
        return self.get_gpu_flops()

    def run_monitor(self, system):
        value = self.get_parameter_value()
        now = time.monotonic()
        if value is not None:
            system.resource_info[self.name] = value
            self._last_success_at = now
            return
        if (
            self._last_success_at is None
            or now - self._last_success_at >= self._last_known_good_seconds
        ):
            system.resource_info.pop(self.name, None)
