# Backend API

The backend service is Dayu's control plane. It is responsible for policy discovery, DAG and datasource management, deployment orchestration, runtime query control, visualization configuration, and log export.

Implementation entrypoint: `backend/backend_server.py`

## Main Data Models

### DAG workflow

`GET /dag_workflow` and `POST /dag_workflow` operate on DAG definitions with the following structure:

```json
{
  "dag_id": 1,
  "dag_name": "car-pipeline",
  "dag": {
    "node_1": {
      "id": "node_1",
      "prev": [],
      "succ": ["node_2"],
      "service_id": "car-detection"
    },
    "node_2": {
      "id": "node_2",
      "prev": ["node_1"],
      "succ": [],
      "service_id": "license-plate-recognition"
    }
  }
}
```

### Datasource configuration

Datasource configs are uploaded as YAML files through `POST /datasource`. A typical config looks like:

```yaml
source_name: "Road & Street Cameras (Two Camera HTTP)"
source_type: "video"
source_mode: "http_video"
source_list:
  - name: "road_camera_0"
    url: ""
    dir: "road_dense/"
    metadata:
      resolution: "540p"
      fps: 30
      encoding: "mp4v"
      buffer_size: 4
```

### Result visualization configuration

Result visualization configs are YAML arrays uploaded through `POST /result_visualization_config/{source_id}`.

```yaml
- name: Frame Visualization
  type: image
  variables: ["Frame with Regions of Interest"]
  hook_name: roi_frame
  size: 1
```

## Endpoint Groups

### Catalog and topology

| Method | Path | Purpose | Request | Response |
| --- | --- | --- | --- | --- |
| `GET` | `/policy` | List available scheduler policies from `template/scheduler_policies.yaml`. | None | Array of `{policy_id, policy_name}` |
| `GET` | `/installed_service` | List logical processor service ids from the committed RuntimeDirectory. | None | Array of service ids |
| `GET` | `/service` | List services declared in `template/services.yaml`. | None | Array of service metadata |
| `GET` | `/service_info/{service}` | Read normalized Pod resources and the shared WAN probe from the cached batch snapshot for one logical service. | Path parameter `service` | Array of service-detail records described below. |
| `GET` | `/edge_node` | List Ready edge nodes from backend's owned inventory snapshot. | None | Array of `{name}` |

### DAG and datasource management

| Method | Path | Purpose | Request | Response |
| --- | --- | --- | --- | --- |
| `GET` | `/dag_workflow` | List all DAG workflows currently stored in backend memory. | None | Array of DAG workflow objects |
| `POST` | `/dag_workflow` | Add or update a DAG workflow. | JSON body with `dag_name` and `dag` | `{state, msg}` |
| `DELETE` | `/dag_workflow` | Delete a DAG workflow by id. | JSON body with `dag_id` | `{state, msg}` |
| `GET` | `/datasource` | List uploaded datasource configurations. | None | Array of datasource config objects |
| `POST` | `/datasource` | Upload one or more datasource YAML config files. | `multipart/form-data` with `file` or files | `{state, msg, results}` |
| `DELETE` | `/datasource` | Delete a datasource config by label. | JSON body with `source_label` | `{state, msg}` |

### Deployment and query lifecycle

| Method | Path | Purpose | Request | Response |
| --- | --- | --- | --- | --- |
| `POST` | `/install` | Resolve policy + datasource mapping and deploy the runtime stack. | JSON body described below | `{state, msg}` |
| `POST` | `/stop_service` | Accept an asynchronous uninstall: close query admission, persist `uninstalling`, and start background teardown. | None | `{state, msg}`; success means accepted, not completed |
| `GET` | `/install_state` | Check session ownership and its lifecycle phase; failed/recovering sessions remain `install` until safely uninstalled. | None | `{state, phase, ready, operation_id, active_directory_revision, active_runtime_count, pending_runtime_count, cleanup_runtime_count, retirement_revision, retirement_deadline, last_error}` |
| `POST` | `/submit_query` | Open datasource playback for a datasource label and begin result collection. | JSON body with `source_label` | `{state, msg}` |
| `POST` | `/stop_query` | Stop datasource playback and clear in-memory task results. | None | `{state, msg}` |
| `GET` | `/query_state` | Get query state for the current datasource. | None | `{state: "open"|"close"|"disabled", source_label}` |
| `GET` | `/source_list` | List active source ids and labels for the currently opened datasource. | None | Array of `{id, label}` |
| `GET` | `/datasource_state` | Return the datasource supervisor view of the current datasource state. | None | `{state: "open"|"close", ...config}` |
| `POST` | `/reset_datasource` | Cancel the active result-collector generation and clear datasource state. | None | `null` |

`state="install"` means a managed session still owns resources; it is not a readiness signal. Clients may query
`/installed_service`, `/service_info/{service}`, and `/system_parameters` only when `phase="active"`. During activation,
publication recovery, or uninstall, the frontend retains the ownership state to prevent a second install while
suspending telemetry polling and clearing active-service details. A pending old-revision retirement keeps the newly
published session active and does not block those reads.

`ready` is true exactly when `phase="active"`. With no session, the endpoint returns
`state="uninstall"`, `phase="uninstalled"`, zero counts/revision, and an empty error. The lifecycle request itself may
still be running in a worker thread; this endpoint remains responsive throughout activation, publication recovery,
background retirement reconciliation, or exact-UID deletion.

After `POST /stop_service` returns `state="success"`, poll `/install_state`. The uninstall is complete only when the
session disappears and the endpoint reports `state="uninstall", phase="uninstalled"`. Until then, `operation_id`
identifies the accepted operation, `phase` is `uninstalling` or `finalizing-uninstall`, and `last_error` reports the
latest background failure while the reconcile worker keeps the exact ownership record for retry.

`POST /install` expects a deployment request shaped like:

```json
{
  "source_config_label": "Road & Street Cameras (Two Camera HTTP)",
  "policy_id": "casva",
  "source": [
    {
      "id": 0,
      "name": "road_camera_0",
      "dag_selected": 1,
      "node_selected": ["edgex1", "edgex2"]
    }
  ]
}
```

During install, backend creates immutable `sedna.io/v1alpha1` `RuntimeService` objects. Scheduler is activated first,
then supplies source-selection and initial-deployment plans. Backend validates the deployment plan and, when
`default-cloud-processor-backup` is enabled, adds the resolved cloud node to every logical processor placement. It
activates the remaining workers and publishes RuntimeDirectory revision 1 only after all exact identities are Ready
and Activated.

The `dayu-runtime-session` ConfigMap is the compact transaction record. Its `resourceVersion` is the compare-and-swap
token for lifecycle changes. Its normalized `source_deploy` preserves both processor `node_set` and the exact
`source_candidate_nodes`/`source_selection_scope` authorization used for the install. Runtime routing is authoritative
in Scheduler's committed RuntimeDirectory; no local YAML file participates in install, redeploy, or uninstall.

### Runtime telemetry cost

When an active RuntimeDirectory is committed, Backend binds all processor Pod names/UIDs and their logical-service
context into one generation-scoped telemetry worker. Every due Kubernetes cycle performs one
server-side-label-filtered namespaced Pod list and, when available, one equally filtered metrics API list for the whole
directory, joins by exact UID, and reuses the independently cached node inventory. The default Kubernetes cadence is
10 seconds rather than the browser or Scheduler cadence.

`GET /service_info/{service}` only filters and deep-copies that in-memory snapshot. It performs no lifecycle lookup,
Scheduler request, or Kubernetes call. Rebind/uninstall clears processor metrics immediately, and an in-flight sample
from an older install or directory revision cannot publish into the new generation. Immediately after bind, exact
processor routes appear as placeholders with their committed target node and unknown resource/readiness fields; the
first successful batch sample atomically replaces them. A failed sample retains the placeholder or prior
last-known-good batch. Runtime Pods neither serve this request nor call Kubernetes.

CPU and memory are sums across every expected container reported for the exact active Pod. The denominator prefers the
target Node's `status.allocatable` and falls back to `status.capacity` only when allocatable is absent or invalid;
`basis` always exposes which denominator was used. CPU is returned in millicores and memory in bytes so clients do not
need to parse Kubernetes Quantity strings. Missing, partial, malformed, or temporarily failed metrics are never
converted to zero. They are exposed as `collecting`, `unavailable`, or a retained `stale` value.

```json
[
  {
    "ip": "10.0.0.8",
    "hostname": "edge-a",
    "cpu": {
      "status": "available",
      "usage_millicores": 25.0,
      "reference_millicores": 4000.0,
      "utilization_percent": 0.625,
      "basis": "node_allocatable"
    },
    "memory": {
      "status": "available",
      "usage_bytes": 67108864,
      "reference_bytes": 8589934592,
      "utilization_percent": 0.78125,
      "basis": "node_allocatable"
    },
    "bandwidth": {
      "status": "available",
      "mbps": 12.34,
      "probe_node": "edge-probe"
    },
    "age": "2026-07-12T00:00:00Z"
  }
]
```

`bandwidth` is deliberately a shared system measurement, not a per-processor-node link estimate. One edge Monitor owns
the Scheduler's `available_bandwidth` lock and runs the edge-to-cloud iperf client; the cloud server and non-holder
nodes publish `-1`, while probe failure publishes `0`. Backend projects the only finite positive sample to every
service row and identifies its `probe_node`. No valid sample gives `collecting` or `unavailable`; a retained sample
after a Scheduler resource-read failure is `stale`; multiple positive candidates fail closed as `ambiguous`.

The list selector is Sedna's controller-guaranteed `dayu.io/mesh-managed=true`; exact Pod name/UID matching remains the
authorization boundary. The application-supplied `app.kubernetes.io/managed-by` label owns RuntimeService CRs but is
not trusted as the materialized-Pod identity contract.

### Runtime data, visualization, and logs

| Method | Path | Purpose | Request | Response |
| --- | --- | --- | --- | --- |
| `GET` | `/task_result` | Fetch visualization-ready task data for each active source. | None | Object keyed by `source_id` |
| `GET` | `/system_parameters` | Fetch one system visualization snapshot and append it to the backend log store. | None | Array with one `{timestamp, data}` snapshot |
| `GET` | `/result_visualization_config/{source_id}` | Get result visualization config for one source. | Path parameter `source_id` | Array of visualization config objects with generated `id` |
| `POST` | `/result_visualization_config/{source_id}` | Upload a source-specific result visualization config file. | `multipart/form-data` with `file` | `{state, msg}` |
| `GET` | `/system_visualization_config` | Get system visualization config. | None | Array of visualization config objects with generated `id` |
| `GET` | `/download_log` | Stream exported result logs from distributor as a `.json.gz` download. | None | `application/gzip` stream |
| `GET` | `/download_system_log` | Export system visualization snapshots as a JSON file. | None | File download |

## Response Notes

### `/task_result`

The response is grouped by source id. Each source contains recent task outputs, and each task contains visualization data already transformed through result-visualizer hooks.

```json
{
  "0": [
    {
      "task_id": 12,
      "data": [
        {
          "id": 0,
          "data": {
            "Frame with Regions of Interest": "<base64-image>"
          }
        }
      ]
    }
  ]
}
```

### `/system_parameters`

One backend-owned worker fetches Scheduler resource/overhead and due Kubernetes runtime metrics independently of
browser traffic. It permits only one sampling cycle at a time, applies separate bounded timeout/cadence settings, and
retains each last-known-good field across transient failures. This endpoint performs no Scheduler or Kubernetes I/O: it
renders the cached values, stores the rendered snapshot in `system_log_store.jsonl`, and returns it in a single-element
array. Before the first successful sample, visualizers use their existing empty/default values.

```json
[
  {
    "timestamp": "12:30:15",
    "data": [
      {
        "id": 0,
        "data": {
          "edgex1": 38.4,
          "cloudx1": 62.1
        }
      }
    ]
  }
]
```

## Behavioral Notes

- `POST /install` treats each source `node_selected`/`node_set` as processor candidates. A policy with
  `scope=all_edge_nodes` receives a separate Backend-authorized source set and may select a generator node outside the processor set;
  `scope=selected_edge_nodes` uses the processor set for both roles.
- `POST /install` fails closed if any required processor/cloud target lacks a Ready Sedna LC/EdgeMesh agent, if a fixed
  source is outside the resolved source permission set, if activation identity is
  incomplete, or if RuntimeDirectory readback differs from the published revision/hash.
- `POST /stop_service` is idempotent admission for a background operation and never waits for task leases or resource
  deletion. Backend subsequently deletes generators first, sends
  `deadline=now` retirement fences for every possibly published revision, and attempts the install-id-guarded
  RuntimeDirectory/proposal clear. Fence or clear failure is logged and does not veto administrative teardown. Backend
  then persists `finalizing-uninstall`, deletes Scheduler by exact UID as the definitive admission fence, deletes the
  remaining workers by exact UID, and removes `dayu-runtime-session` after every UID-guarded `Background` DELETE is
  accepted by the apiserver or the exact target is already absent. It does not wait for dependent objects to disappear
  physically. An API rejection or timeout preserves the session phase and error for retry; retrying
  `finalizing-uninstall` does not require a live Scheduler API.
  If install is still activating or planning, stop first cancels its lifecycle token and then enters serialized
  cleanup. Pre-publication cleanup uses the persisted install identity and exact UID discovery without calling an
  unready Scheduler; cancellation is not persisted as a generic install failure. Concurrent stop requests keep new
  install admission closed until every stop request has finished.
- `POST /datasource` returns `state: "partial"` when only some uploaded files are accepted. The `results` array carries per-file status details.
- `POST /submit_query` only works after install-time deployment has completed and a datasource config exists for the requested label.
- `POST /submit_query` is idempotent for the already-open datasource label. Opening a different datasource while one is active fails until `/stop_query` closes the current one.
- Query startup is immediate; result collection uses the configured bounded request timeout and batch size rather than an unbounded Distributor response.
- `POST /stop_query` is idempotent and succeeds when the datasource is already closed.
- `GET /query_state` returns `disabled` when `template/base.yaml` sets `datasource.use-simulation=false`.
- `GET /source_list`, `GET /task_result`, and `GET /datasource_state` are runtime-state dependent. They return empty collections or closed state when no datasource is active.
- Visualization config upload uses YAML validation in backend memory. The file is accepted only if each visualization entry contains a valid `name`, `type`, `variables`, and `size`, plus valid hook metadata when present.
