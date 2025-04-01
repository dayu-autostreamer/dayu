def import_application():
    import os
    from importlib import import_module

    service_name = os.getenv('SERVICE_NAME')
    if service_name and service_name.startswith('processor-'):
        processor = service_name[len('processor-'):].replace('-', '_')
        module = import_module(f"core.applications.{processor}")

        from core.lib.common import config
        context_namespace = config.__dict__
        for attr in getattr(module, "__all__", []):
            context_namespace[attr] = getattr(module, attr)


import_application()
