import kubernetes as k8s
from core.lib.common import Context, SystemConstant


class PortInfo:

    @staticmethod
    def get_component_port(component_name: str) -> int:
        ports_dict = PortInfo.get_all_ports(component_name)
        ports_list = list(ports_dict.values())
        if ports_list:
            return ports_list[0]
        raise Exception(f"Component '{component_name}' does not exist.")

    @staticmethod
    def get_all_ports(keyword: str) -> dict:
        ports_dict = {}
        k8s.config.load_incluster_config()
        v1 = k8s.client.CoreV1Api()
        namespace = Context.get_parameter('NAMESPACE')
        svcs = v1.list_namespaced_service(namespace)
        for svc in svcs.items:
            if keyword in svc.metadata.name:
                if svc.spec.type != "NodePort":
                    assert None, f"Service '{svc.metadata.name}' is not of type NodePort."
                ports_dict[svc.metadata.name] = int(svc.spec.ports[0].node_port)
        return ports_dict

    @staticmethod
    def get_service_ports_dict() -> dict:
        component_name = SystemConstant.PROCESSOR.value
        PortInfo.get_all_ports(component_name)
        ports_dict = PortInfo.get_all_ports(component_name)
        component_ports_dict = {}
        for svc_name in ports_dict:
            # get sub service name
            des_name = '-'.join(svc_name.split('-')[1:-1])
            component_ports_dict[des_name] = ports_dict[svc_name]

        return component_ports_dict

    @staticmethod
    def get_service_port(service_name: str) -> int:
        return PortInfo.get_service_ports_dict().get(service_name)
