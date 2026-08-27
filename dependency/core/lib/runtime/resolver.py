"""Exact task-route resolution with no external discovery or refresh path."""

from .context import RuntimeContext
from .model import RuntimeEndpoint


class RuntimeResolver:
    TASK_ROUTED_COMPONENTS = frozenset({"controller", "processor"})

    def __init__(self, context=None):
        self.context = context or RuntimeContext.get_default()

    @staticmethod
    def _task_routes(task):
        if task is None:
            return None
        getter = getattr(task, "get_runtime_routes", None)
        if callable(getter):
            return getter()
        if isinstance(task, dict):
            return task.get("runtime_routes")
        return None

    @classmethod
    def list_routes(cls, task_or_routes, component=None, target_node=None, logical_service=None):
        routes = cls._task_routes(task_or_routes)
        if routes is None and isinstance(task_or_routes, (dict, list)):
            routes = task_or_routes
        if isinstance(routes, dict) and "routes" in routes:
            routes = routes.get("routes")
        endpoints = RuntimeContext._normalize_endpoints(routes or {})
        return [
            endpoint for endpoint in endpoints
            if endpoint.matches(component, target_node, logical_service)
        ]

    @staticmethod
    def _select_unique(matches, component, target_node, logical_service):
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            # Repeated copies of exactly the same identity/address are harmless;
            # conflicting answers are not.
            identities = {
                (item.runtime_id, item.runtime_service_uid, item.service_uid, item.endpoint_pod_uid,
                 item.fqdn, item.port, item.deployment_revision)
                for item in matches
            }
            if len(identities) == 1:
                return matches[0]
            raise ValueError(
                "ambiguous task runtime route: component={!r}, service={!r}, node={!r}".format(
                    component, logical_service, target_node
                )
            )
        return None

    def resolve(self, component, task=None, target_node=None, logical_service=None, exact=False, required=True):
        matches = self.list_routes(task, component, target_node, logical_service)
        endpoint = self._select_unique(matches, component, target_node, logical_service)
        if endpoint:
            if component in self.TASK_ROUTED_COMPONENTS:
                endpoint.validate_exact()
            return endpoint

        # Controller and processor identities must come from the task snapshot;
        # only infrastructure services may use process bootstrap endpoints.
        if component not in self.TASK_ROUTED_COMPONENTS and not exact:
            endpoint = self.context.resolve_static_endpoint(
                component,
                target_node=target_node,
                logical_service=logical_service,
                required=False,
            )
            if endpoint:
                return endpoint

        if required:
            raise LookupError(
                "runtime route missing: component={!r}, service={!r}, node={!r}, exact={!r}".format(
                    component, logical_service, target_node, exact
                )
            )
        return None

    def resolve_url(self, component, path=None, task=None, target_node=None,
                    logical_service=None, exact=False, required=True):
        endpoint = self.resolve(
            component,
            task=task,
            target_node=target_node,
            logical_service=logical_service,
            exact=exact,
            required=required,
        )
        return endpoint.url(path) if endpoint else None
