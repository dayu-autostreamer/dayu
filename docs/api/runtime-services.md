# Runtime Service APIs

This document describes the internal APIs used by Dayu runtime components. These are repository-internal service contracts, not frontend-facing APIs.

## Shared Runtime Contracts

| Topic | Behavior |
| --- | --- |
| Task payload | Internal services exchange serialized `Task` strings produced by `Task.serialize()`. |
| File transfer | Binary task content is sent as `multipart/form-data` with a `file` field plus a `data` field containing the serialized task. |
| Scheduler/resource updates | Scheduler endpoints often receive JSON encoded into a form field named `data`. |
| Compatibility | Several endpoints use `GET` while still reading form or JSON request bodies. Clients in this repository depend on that behavior. |

## Controller Service

Implementation entrypoint: `dependency/core/controller/controller_server.py`

| Method | Path | Purpose | Request | Response |
| --- | --- | --- | --- | --- |
| `POST` | `/check` | Check whether the processor layer is reachable and healthy. | None | `{status: "ok"|"not ok"}` |
| `POST` | `/submit_task` | Accept a new task from generator or another controller. | Multipart with `file` and serialized `data` | Empty `200` response after background enqueue |
| `POST` | `/process_return_task` | Accept a processed task returned by processor. | Form field `data` with serialized task | Empty `200` response after background enqueue |

Operational notes:

- On startup the controller clears its temp directory.
- If `DELETE_TEMP_FILES` is enabled, a `FileCleaner` thread removes stale temp files.
- `submit_task` stores the uploaded file in the temp directory and forwards the task into the controller pipeline asynchronously.

## Processor Service

Implementation entrypoint: `dependency/core/processor/processor_server.py`

| Method | Path | Purpose | Request | Response |
| --- | --- | --- | --- | --- |
| `GET` | `/health` | Basic health probe. | None | `{status: "ok"}` |
| `POST` | `/predict` | Queue a task that includes a file payload. | Multipart with `file` and serialized `data` | Empty `200` response after background enqueue |
| `POST` | `/predict_local` | Queue a task that does not require an uploaded file. | Form field `data` | Empty `200` response after background enqueue |
| `POST` | `/predict_and_return` | Process a task synchronously and return the serialized result. | Multipart with `file` and serialized `data` | Serialized task string or `null` |
| `GET` | `/queue_length` | Return current queue size. | None | Integer |
| `POST` | `/queue_clear` | Preview or remove queued processor tasks. | Form field `data` with JSON such as `{"dry_run": true, "max_count": 10, "reason": "manual_queue_clear"}` | Queue clear summary with matched, cleared, remaining, and dropped task metadata |
| `GET` | `/model_flops` | Return the processor model FLOPs value. | None | Numeric FLOPs value |
| `GET` | `/model_memory` | Return the processor process RSS in bytes. | None | Integer memory usage |

Operational notes:

- `PROCESSOR_NAME` selects the processor implementation.
- `PRO_QUEUE_NAME` selects the queue strategy.
- A background thread drains the task queue and posts results back to controller through `/process_return_task`.
- `/queue_clear` supports dry-run previews when the selected queue exposes `get_all_without_drop()`. Destructive clears
  prefer a queue-level `drain(max_count=...)` method and otherwise fall back to repeated `get()` calls.
- All processor implementations store inference content as `{"service", "outputs", "profile"}`. `outputs` is keyed by
  generic form labels such as `bbox`, `text`, `segmentation`, `track`, `attribute`, `trajectory`, `pose`, or `graph`;
  each label maps to records with `frame_index` and `items`.
- `structured_processor` is a generic processor shell for services whose `Structured_Processor` consumes upstream content envelopes
  and returns only service-specific `outputs`. The processor wraps those outputs with `service` and a compact
  `profile` containing `frame_count`.
- Structured processor services should not encode DAG membership or shared DAG schemas in their outputs. Users compose
  services into DAGs at runtime through the service catalog and workflow definition.

## Scheduler Service

Implementation entrypoint: `dependency/core/scheduler/scheduler_server.py`

| Method | Path | Purpose | Request | Response |
| --- | --- | --- | --- | --- |
| `GET` | `/schedule` | Generate a schedule plan for one source. | Form field `data` with JSON object | `{plan, deployment}` |
| `GET` | `/overhead` | Get average scheduler overhead across agents. | None | Number of seconds |
| `POST` | `/scenario` | Update scheduler state with a processed task scenario. | Form field `data` with serialized task | `null` |
| `POST` | `/resource` | Update scheduler resource table for one device. | Form field `data` with JSON `{"device","resource"}` | `null` |
| `GET` | `/resource` | Get the full scheduler resource table. | None | Object keyed by device |
| `GET` | `/resource_lock` | Acquire resource ownership for a monitor probe such as bandwidth. | Form field `data` with JSON `{"resource","device"}` | `{holder}` |
| `GET` | `/source_nodes_selection` | Generate source-to-edge-node selection plan. | Form field `data` with JSON array | `{plan}` |
| `GET` | `/initial_deployment` | Generate initial deployment plan. | Form field `data` with JSON array | `{plan}` |
| `GET` | `/redeployment` | Generate redeployment plan. | Form field `data` with JSON array | `{plan}` |

`/schedule` expects data close to:

```json
{
  "source_id": 0,
  "meta_data": {"resolution": "720p", "fps": 5, "buffer_size": 4, "encoding": "mp4v"},
  "source_device": "edgex1",
  "all_edge_devices": ["edgex1", "edgex2"],
  "dag": {
    "start": {"service": {"execute_device": "edgex1"}},
    "car-detection": {"service": {"execute_device": "cloudx1"}}
  }
}
```

Different `GEN_BSO` implementations may append scheduler-specific fields such as `skip_count`, `frame`, or `hash_code`.

## Distributor Service

Implementation entrypoint: `dependency/core/distributor/distributor_server.py`

| Method | Path | Purpose | Request | Response |
| --- | --- | --- | --- | --- |
| `POST` | `/distribute` | Persist a finished task and forward scenario data to scheduler. | Multipart with `file` and serialized `data` | Empty `200` response after background processing |
| `GET` | `/result` | Incrementally fetch stored task results. | JSON body `{"size","time_ticket"}` | `{result, time_ticket, size}` |
| `GET` | `/file` | Download a generated file and schedule it for deletion. | JSON body `{"file":"<path>"}` | File response |
| `GET` | `/result_by_time` | Query results for a time range. | JSON body `{"start_time","end_time","source_id?"}` | `{result, size}` |
| `GET` | `/all_result` | Dump all stored results. | None | `{result, size}` |
| `GET` | `/export_result_log` | Export stored results as a gzip-compressed JSON file. | None | `application/gzip` file |
| `POST` | `/clear_database` | Clear the result database. | None | `null` |
| `GET` | `/is_database_empty` | Check whether the result database contains records. | None | Boolean |

Compatibility note:

- `/result`, `/file`, and `/result_by_time` are implemented as `GET` routes but still expect a JSON request body. This is an implementation detail that callers inside the repository currently rely on.

## HTTP Video Source Service

Implementation entrypoint: `datasource/http_video.py`

The HTTP video source service is only used for simulated `http_video` sources. It exposes one admin endpoint plus per-source dynamic routes.

For datasource directory layout, manifest schema, and frame-indexing behavior, see [`../datasource/README.md`](../datasource/README.md).

### Admin route

| Method | Path | Purpose | Request | Response |
| --- | --- | --- | --- | --- |
| `POST` | `/admin/add_source` | Register a new source path and mount its dynamic routes. | JSON `{"root","path","play_mode"}` | `{status}` |

### Dynamic per-source routes

After a source is registered under `path=<source-path>`, two routes become available:

| Method | Path | Purpose | Request | Response |
| --- | --- | --- | --- | --- |
| `GET` | `/<source-path>/source` | Generate the next buffered clip for that source. | Form field `data` with JSON request | JSON array of frame hash or frame index values |
| `GET` | `/<source-path>/file` | Download the clip generated by the previous `/source` call. | None | File response |

The `/source` request JSON includes the generator-selected hook names:

```json
{
  "source_id": 0,
  "task_id": 10,
  "meta_data": {"resolution": "720p", "fps": 10, "buffer_size": 4},
  "raw_meta_data": {"resolution": "1080p", "fps": 30, "buffer_size": 4},
  "gen_filter_name": "simple",
  "gen_process_name": "simple",
  "gen_compress_name": "simple"
}
```

The service resolves those hook names dynamically, applies frame filtering, frame processing, and compression, then returns the generated file through `/<source-path>/file`.

## V4L2 Video Getter

Implementation entrypoint: `dependency/core/lib/algorithms/data_getter/v4l2_video_getter.py`

`v4l2_video` is used for real-camera datasource configs such as `config/datasource_configs/real_camera.yaml`. It is a
generator data getter rather than a datasource-supervisor service, so it does not expose a repository-managed HTTP API
or use the `http_video`/`rtsp_video` manifest layout.

## Internal Non-HTTP Entry Points

These runtime components are important for understanding the system but do not expose repository-managed public HTTP routes:

| Component | Behavior | Main code path |
| --- | --- | --- |
| Generator | Instantiates the configured generator type and starts its run loop. | `dependency/core/generator/generator_server.py` |
| Monitor | Periodically samples `MON_PRAM` hooks and posts resource data to scheduler. | `dependency/core/monitor/monitor.py` |
| Datasource supervisor | Polls backend `/datasource_state` and starts or stops local source processes. | `datasource/datasource_server.py` |
| RTSP stream source | Reads `rtsp_video/manifest.json` through `VideoDataset` and streams clips to the configured RTSP address. | `datasource/rtsp_video.py` |
| Video dataset loader | Loads manifest-driven clip order, `video_root`, and frame-index metadata for both `http_video` and `rtsp_video`. | `datasource/video_dataset.py` |
| V4L2 getter | Opens a local camera device from `source.url` and creates generator tasks directly. | `dependency/core/lib/algorithms/data_getter/v4l2_video_getter.py` |
