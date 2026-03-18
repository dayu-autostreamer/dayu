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
| `GET` | `/installed_service` | List currently installed service ids based on running pods. | None | Array of service ids |
| `GET` | `/service` | List services declared in `template/services.yaml`. | None | Array of service metadata |
| `GET` | `/service_info/{service}` | Get runtime information for a deployed service. | Path parameter `service` | Service-specific runtime info from Kubernetes |
| `GET` | `/edge_node` | List known edge nodes. | None | Array of edge node descriptors |

### DAG and datasource management

| Method | Path | Purpose | Request | Response |
| --- | --- | --- | --- | --- |
| `GET` | `/dag_workflow` | List all DAG workflows currently stored in backend memory. | None | Array of DAG workflow objects |
| `POST` | `/dag_workflow` | Add or update a DAG workflow. | JSON body with `dag_name` and `dag` | `{state, msg}` |
| `DELETE` | `/dag_workflow` | Delete a DAG workflow by id. | JSON body with `dag_id` | `{state, msg}` |
| `GET` | `/datasource` | List uploaded datasource configurations. | None | Array of datasource config objects |
| `POST` | `/datasource` | Upload a datasource YAML config file. | `multipart/form-data` with `file` | `{state, msg}` |
| `DELETE` | `/datasource` | Delete a datasource config by label. | JSON body with `source_label` | `{state, msg}` |

### Deployment and query lifecycle

| Method | Path | Purpose | Request | Response |
| --- | --- | --- | --- | --- |
| `POST` | `/install` | Resolve policy + datasource mapping and deploy the runtime stack. | JSON body described below | `{state, msg}` |
| `POST` | `/stop_service` | Uninstall deployed runtime components. | None | `{state, msg}` |
| `GET` | `/install_state` | Check whether the stack is installed. | None | `{state: "install"|"uninstall"}` |
| `POST` | `/submit_query` | Open datasource playback for a datasource label and begin result collection. | JSON body with `source_label` | `{state, msg}` |
| `POST` | `/stop_query` | Stop datasource playback and clear in-memory task results. | None | `{state, msg}` |
| `GET` | `/query_state` | Get query state for the current datasource. | None | `{state, source_label}` |
| `GET` | `/source_list` | List active source ids and labels for the currently opened datasource. | None | Array of `{id, label}` |
| `GET` | `/datasource_state` | Return the datasource supervisor view of the current datasource state. | None | `{state: "open"|"close", ...config}` |
| `POST` | `/reset_datasource` | Force datasource state to closed in backend memory. | None | `null` |

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

The backend fetches scheduler resource data once, renders the configured system visualizers, stores a snapshot in `system_log_store.jsonl`, and returns the latest snapshot in a single-element array.

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

## Compatibility Notes

- `POST /install` assumes the requested scheduler policy, datasource configuration, DAG workflow, and edge nodes all already exist and are mutually compatible.
- `POST /submit_query` only works after install-time deployment has completed and a datasource config exists for the requested label.
- `GET /source_list`, `GET /task_result`, and `GET /datasource_state` are runtime-state dependent. They return empty collections or closed state when no datasource is active.
- Visualization config upload uses YAML validation in backend memory. The file is accepted only if each visualization entry contains a valid `name`, `type`, `variables`, and `size`, plus valid hook metadata when present.
