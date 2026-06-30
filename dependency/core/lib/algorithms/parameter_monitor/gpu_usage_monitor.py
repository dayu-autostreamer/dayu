import abc
import time

from .base_monitor import BaseMonitor

from core.lib.common import ClassFactory, ClassType, LOGGER

__all__ = ('GPUUsageMonitor',)


@ClassFactory.register(ClassType.MON_PRAM, alias='gpu_usage')
class GPUUsageMonitor(BaseMonitor, abc.ABC):
    PEAK_HOLD_HALF_LIFE_S = 5.0
    MIN_REPORTED_RATIO = 0.01

    def __init__(self, system):
        super().__init__(system)
        self.name = 'gpu_usage'
        self._smoothed_ratio = 0.0
        self._last_sample_ts = None

    def get_parameter_value(self):
        """Return recent GPU pressure as a ratio in [0, 1].

        Backend helpers still report utilization in percent (0-100). This method
        normalizes the first successful backend result into a ratio so the
        scheduler and Hedger can consume a consistent unit. GPU inference is
        often bursty, so a single instantaneous sample can miss active work and
        report many zeros. The returned value therefore keeps a decayed recent
        peak, making the signal closer to "recent GPU pressure" than a raw point
        sample.

        Fallback order:
        1) NVML via pynvml
        2) nvidia-smi CLI
        3) Jetson sysfs devfreq load files
        4) tegrastats one-line sample
        If none available or no GPU present, return 0.0.
        """
        percent = self._read_instantaneous_percent()
        if percent is None:
            LOGGER.warning('[GPUUsage] Unable to determine GPU usage, returning 0.0.')
            ratio = 0.0
        else:
            ratio = max(0.0, min(100.0, float(percent))) / 100.0
        return self._smooth_recent_peak(ratio)

    def _read_instantaneous_percent(self):
        """Read one instantaneous backend sample in percent, or None."""
        try:
            percent = self._get_usage_via_nvml()
            if percent is not None:
                return percent
        except Exception:
            pass

        try:
            percent = self._get_usage_via_nvidia_smi()
            if percent is not None:
                return percent
        except Exception:
            pass

        try:
            percent = self._get_usage_via_jetson_sysfs()
            if percent is not None:
                return percent
        except Exception:
            pass

        try:
            percent = self._get_usage_via_tegrastats()
            if percent is not None:
                return percent
        except Exception:
            pass
        return None

    def _smooth_recent_peak(self, ratio: float) -> float:
        """Keep a decayed recent peak so short GPU bursts remain visible."""
        now = time.monotonic()
        ratio = max(0.0, min(1.0, float(ratio)))
        if self._last_sample_ts is None:
            smoothed = ratio
        else:
            dt = max(0.0, now - self._last_sample_ts)
            decay = 0.5 ** (dt / self.PEAK_HOLD_HALF_LIFE_S)
            smoothed = max(ratio, self._smoothed_ratio * decay)
        if smoothed < self.MIN_REPORTED_RATIO:
            smoothed = 0.0
        self._smoothed_ratio = smoothed
        self._last_sample_ts = now
        return smoothed

    @staticmethod
    def _get_usage_via_nvml():
        try:
            import pynvml  # nvidia-ml-py3
        except Exception:
            return None
        try:
            pynvml.nvmlInit()
            try:
                count = pynvml.nvmlDeviceGetCount()
                if count == 0:
                    return 0
                utils = []
                for i in range(count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    # rates.gpu is percent 0-100
                    utils.append(int(getattr(rates, 'gpu', 0)))
                return max(utils) if utils else 0
            finally:
                try:
                    pynvml.nvmlShutdown()
                except Exception:
                    pass
        except Exception:
            return None

    @staticmethod
    def _get_usage_via_nvidia_smi(timeout_sec: float = 1.0):
        import subprocess
        import shutil
        if shutil.which('nvidia-smi') is None:
            return None
        try:
            # Query GPU utilization for all devices, values without header and units
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout_sec, check=False, text=True
            )
            if result.returncode != 0:
                return None
            lines = [ln.strip() for ln in result.stdout.strip().splitlines() if ln.strip()]
            utils = []
            for ln in lines:
                try:
                    utils.append(int(ln))
                except ValueError:
                    continue
            if not utils:
                return None
            return max(0, min(100, max(utils)))
        except Exception:
            return None

    @staticmethod
    def _get_usage_via_jetson_sysfs():
        # Try common Jetson load files; values are often in per-mille (0..1000) or 0..255 scale
        import os
        import glob
        candidates = [
            '/sys/devices/gpu.0/load',
            '/sys/devices/*gpu*/load',
            '/sys/devices/*gv11b*/load',
            '/sys/devices/platform/*gpu*/load',
            '/sys/devices/platform/host1x/*gpu*/load',
            '/sys/devices/gpu.0/devfreq/*/load',
            '/sys/devices/*gpu*/devfreq/*/load',
            '/sys/devices/*gv11b*/devfreq/*/load',
            '/sys/devices/*nvgpu*/devfreq/*/load',
            '/sys/class/devfreq/*gpu*/load',
            '/sys/class/devfreq/*nvgpu*/load',
        ]
        paths = []
        for pattern in candidates:
            paths.extend(glob.glob(pattern))
        # Deduplicate while preserving order
        seen = set()
        unique_paths = []
        for p in paths:
            if p not in seen and os.path.isfile(p):
                seen.add(p)
                unique_paths.append(p)
        vals = []
        for p in unique_paths:
            try:
                with open(p, 'r') as f:
                    raw = f.read().strip()
                if not raw:
                    continue
                val = int(''.join(ch for ch in raw if ch.isdigit()))
                # Heuristic scaling
                if val <= 100:
                    percent = float(val)  # Already percent
                elif val <= 1000:
                    percent = val / 10.0  # Per-mille to percent (10*percentage)
                elif val <= 255:
                    percent = val / 255.0 * 100.0 # Scale 0..255 to percent
                else:
                    percent = val / 10.0
                vals.append(percent)
            except Exception:
                continue
        if not vals:
            return None
        percent = max(vals)
        return max(0.0, min(100.0, percent))

    @staticmethod
    def _get_usage_via_tegrastats():
        import subprocess
        import shutil
        import re
        if shutil.which('tegrastats') is None:
            return None
        line = ''
        try:
            # Run tegrastats and grab the first line, then terminate
            proc = subprocess.Popen(['tegrastats'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                line = proc.stdout.readline().strip() if proc.stdout else ''
            finally:
                try:
                    proc.terminate()
                except Exception:
                    pass
            if not line:
                return None
            m = re.search(r'GR3D_FREQ\s+(\d+)%', line)
            if not m:
                return None
            percent = int(m.group(1))
            return max(0, min(100, percent))
        except Exception:
            return None
