"""Empirical FPS and resolution accuracy models used by STEADY."""

import math

resolution_wh = {
    "240p": {"w": 320, "h": 240},
    "360p": {"w": 640, "h": 360},
    "480p": {"w": 640, "h": 480},
    "540p": {"w": 960, "h": 540},
    "600p": {"w": 800, "h": 600},
    "720p": {"w": 1280, "h": 720},
    "900p": {"w": 1440, "h": 900},
    "1080p": {"w": 1920, "h": 1080},
}


class AccuracyPrediction2fps:
    def predict(self, service_name, service_conf, obj_size=None, obj_speed=None):
        if "detection" not in service_name:
            return 1
        if obj_speed is None or obj_speed == -1:
            a, b, c = 0.97, -1.20, -0.78
        elif obj_speed <= 260:
            a, b, c = 0.98, -0.75, -0.85
        elif obj_speed <= 520:
            a, b, c = 0.97, -1.20, -0.78
        elif obj_speed <= 780:
            a, b, c = 0.985, -0.9, -0.29
        else:
            a, b, c = 1.0, -0.84, -0.18
        return max(0, a + b * math.exp(c * service_conf["fps"]))


class AccuracyPrediction2reso:
    def predict(self, service_name, service_conf, obj_size=None, obj_speed=None):
        if "detection" not in service_name:
            return 1
        if obj_size is None or obj_size == -1:
            a, b, c = 0.99, -0.47, -0.008
        elif obj_size == 0:
            a, b, c = 0.99, -0.2, -0.008
        elif obj_size <= 50000:
            a, b, c = 0.98, -0.63, -0.006
        elif obj_size <= 100000:
            a, b, c = 0.99, -0.47, -0.008
        else:
            a, b, c = 0.99, -0.2, -0.008
        height = resolution_wh[service_conf["resolution"]]["h"]
        return max(0, a + b * math.exp(c * (height - 350)))
