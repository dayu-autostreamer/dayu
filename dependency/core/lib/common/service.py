import re

class ServiceConfig:
    SERVICE_PATTERN = re.compile(r"^processor-(.+?)-[^-]+-(?:cloudworker|edgeworker)-")

    @classmethod
    def map_pod_name_to_service(cls, pod_name: str) -> str:
        match = cls.SERVICE_PATTERN.match(pod_name)
        if match:
            return match.group(1)
        else:
            return None
