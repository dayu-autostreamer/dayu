# Operations Guide

This guide summarizes the repository-managed operational flow. It is not a full cluster installation tutorial; it
documents what the current Dayu code and scripts do once a Kubernetes/KubeEdge/Sedna environment exists. For first-time
preparation and tutorial flow, use the [project documentation site](https://dayu-autostreamer.github.io/docs/).

## Deployment Layers

Dayu has two deployment layers:

| Layer | Started by | Main resources |
| --- | --- | --- |
| Platform support layer | [`dayu.sh`](../../dayu.sh) | namespace, service account, Redis, backend, frontend, optional datasource supervisor |
| Application runtime layer | backend `POST /install` | generator, scheduler, controller, distributor, monitor, and per-service processors |

Keeping these layers separate is important. `dayu.sh ACTION=start` makes the UI/backend available. The user still needs
to select a policy, datasource mapping, DAG, and nodes before backend `/install` deploys the runtime stack.

## `dayu.sh` Inputs

`dayu.sh` reads `TEMPLATE/base.yaml` after expanding repository-local `!include` files.

Key values consumed by the script:

| Field | Runtime effect |
| --- | --- |
| `namespace` | Namespace created for Dayu resources. Official namespaces such as `kube-system`, `kubeedge`, and `sedna` are rejected. |
| `pod-permission.*` | Service account and cluster-role binding used by Dayu containers. |
| `crd-meta.*` | API version and kind for Sedna `JointMultiEdgeService` resources. |
| `default-image-meta.*` | Image registry, repository, and tag used for support-layer containers. |
| `default-file-mount-prefix` | Host-path prefix used for temporary and mounted files. |
| `datasource.*` | Whether a simulated datasource pod is created, where it runs, and where it finds dataset files. |
| `log-export.system.*` | Backend system-log retention and compaction environment variables. |

Common commands:

```bash
TEMPLATE=template ACTION=start bash dayu.sh
TEMPLATE=template ACTION=stop bash dayu.sh
```

## Install Lifecycle

Backend `/install` receives:

- `source_config_label`
- `policy_id`
- per-source `dag_selected`
- per-source `node_selected`

Backend then:

1. finds the scheduler policy in `template/scheduler_policies.yaml`
2. finds the datasource config in backend memory
3. validates that selected DAG ids and edge nodes exist
4. loads component templates and service templates
5. asks scheduler hooks for source selection and processor deployment plans
6. renders `JointMultiEdgeService` documents
7. labels and annotates runtime resources
8. applies them through `KubeHelper`
9. stores an install snapshot in the `dayu-runtime-install-state` ConfigMap

The local `resources.yaml` file is a cache and diagnostic artifact. It is not the only uninstall source of truth.

## Processor Deployment

Processors are generated per logical service and target node. `backend/template_helper.py` can generate:

- one cloud-only processor CR named `processor-{service}-{cloud-node}`
- one edge-only processor CR per selected edge node named `processor-{service}-{edge-node}`

`template/base.yaml` controls default cloud backup behavior:

| `default-cloud-processor-backup` | Meaning |
| --- | --- |
| `true` | Create one cloud-side backup processor for each logical service by default. |
| `false` | Create a cloud processor only when the scheduler plan explicitly selects the real cloud hostname. |

This flag does not mean "never deploy on cloud." An exact scheduler-selected cloud hostname is still honored.

## Redeployment

Runtime redeployment is processor-scoped. `BackendCore.run_cycle_deploy()` periodically asks
`TemplateHelper.finetune_yaml_parameters(..., scopes=['processor'])` for refreshed processor documents.

When redeployment succeeds:

- the same install id is reused
- runtime labels and source/policy annotations are preserved
- the local manifest cache is refreshed
- the install-record ConfigMap is updated

`REDEPLOYMENT_REQUEST_INTERVAL` controls the interval. `0` disables the sleep delay between redeployment cycles but does
not by itself start or stop the redeployment loop.

## Query Lifecycle

`POST /submit_query` opens one datasource label. The call is idempotent for the already-open label and fails if another
label is already open.

`POST /stop_query` closes the active datasource, clears in-memory task result queues, and clears source-specific custom
visualization configs. It is idempotent when the datasource is already closed.

When `datasource.use-simulation=false`, backend automatically opens the selected datasource after a successful install.

## Uninstall and Cleanup

Backend `POST /stop_service` is the graceful runtime uninstall path. It discovers runtime resources by labels and falls
back through:

1. live Kubernetes resources matching `dayu.io/runtime-scope=installation`
2. backend-local `resources.yaml`
3. install-record ConfigMap `dayu-runtime-install-state`

The uninstall path keeps state until deletion succeeds. This makes failed uninstall attempts retryable.

`dayu.sh ACTION=stop` first tries backend `/stop_service` when application runtime resources are present, then removes
remaining Dayu custom resources, services, endpoints, workloads, ConfigMaps, Secrets, service-account bindings, and the
namespace.

## Useful Checks

```bash
kubectl get pods -n dayu
kubectl get svc -n dayu
kubectl get jointmultiedgeservice -n dayu
kubectl get jointmultiedgeservice -n dayu -l dayu.io/runtime-scope=installation
kubectl get cm -n dayu dayu-runtime-install-state -o yaml
```

When debugging source playback, also inspect:

- backend `/query_state`
- backend `/datasource_state`
- datasource config uploaded through `/datasource`
- source dataset manifests described in [`../datasource/README.md`](../datasource/README.md)

When debugging visualization, inspect:

- `template/result-visualizations.yaml`
- `template/system-visualizations.yaml`
- any uploaded `config/visualization_configs/*.yaml`
- hook aliases in [`../hooks/catalog.md`](../hooks/catalog.md)
