import re

class ServiceConfig:
    SERVICE_PATTERN = [
        re.compile(r"^processor-(?P<service>.+)-(?P<node>[^-]+)-(?:edgeworker|cloudworker)(?:-|$)"),
        re.compile(r"^processor-(?P<service>.+)-(?P<node>[^-]+)-[^-]+$"),
        re.compile(r"^processor-(?P<service>.+)-[^-]+$")
        ]


    @classmethod
    def map_pod_name_to_service(cls, pod_name: str) -> str:
        for pattern in cls.SERVICE_PATTERN:
            match = pattern.match(pod_name)
            if match:
                return match.group("service")
        return None
