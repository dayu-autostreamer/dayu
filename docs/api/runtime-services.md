# Runtime Service APIs

This document describes the internal APIs used by Dayu runtime components. These are repository-internal service contracts, not frontend-facing APIs.

## Shared Runtime Contracts

| Topic | Behavior |
| --- | --- |
| Task payload | Internal services exchange serialized `Task` strings produced by `Task.serialize()`. |
| File transfer | Binary task content is sent as `multipart/form-data` with a `file` field plus a `data` field containing the serialized task. |
| Scheduler/resource updates | Scheduler endpoints often receive JSON encoded into a form field named `data`. |
| Runtime bootstrap | `DAYU_RUNTIME_BOOTSTRAP` supplies immutable install context and static infrastructure endpoints; it is never a Kubernetes discovery cache. |
| Exact task routing | `Task` serializes `runtime_directory_revision`, `runtime_routes`, and `root_uuid`. Controller/processor routes must be complete exact identities. |
| Task ownership and retirement | Runtime services acquire/renew/release a Scheduler lease keyed by `(runtime_directory_revision, root_uuid)`; Backend may place one immutable deadline on an old revision. |

Runtime service code does not import Kubernetes, load a kubeconfig, or discover Pods, Nodes, or Services. Missing or
ambiguous task routes fail closed.

## Controller Service

Implementation entrypoint: `dependency/core/controller/controller_server.py`

| Method | Path | Purpose | Request | Response |
| --- | --- | --- | --- | --- |
| `POST` | `/check` | Check processors referenced by exact local runtime routes. | Optional form `data` with JSON containing `runtime_routes` | `{status: "ok"|"not ok"}` |
| `POST` | `/submit_task` | Accept a new task from generator or another controller. | Multipart with `file` and serialized `data` | Empty `200` response after background enqueue |
| `POST` | `/process_return_task` | Accept a processed task returned by processor. | Form field `data` with serialized task | Empty `200` response after background enqueue |
| `POST` | `/processor_queues_clear` | Fan out a dry-run or destructive queue-clear request through exact local processor routes. | Form `data` with `{runtime_routes,services?,timeout_s?,reason?,max_count?,dry_run?}` | Aggregate `{ok,device,service_count,matched_count,cleared_count,remaining_count,services}` |

Operational notes:

- The FastAPI lifespan owns temp-file cleanup: it clears the temp directory at startup and shutdown, and is the only
  place that starts or stops a `FileCleaner`.
- If `DELETE_TEMP_FILES` is enabled, that single lifespan-managed cleaner uses the task-lease TTL, rather than a fixed
  short timeout, so a legitimate long-running processor cannot lose its controller-side file mid-task.
- `submit_task` stores the uploaded file in the temp directory and forwards the task into the controller pipeline asynchronously.
- Controller resolves downstream controller/processor targets only from the Task's exact route snapshot and renews the
  Task lease as work advances. Missing route or failed renewal stops forwarding.

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
- Processor validates the exact task route and keeps its lease renewed throughout execution; it does not resolve
  service placement from local or cluster state. A full TTL without a confirmed heartbeat fails closed.
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
| `GET` | `/schedule` | Generate a schedule plan plus exact routes for one source. | Form field `data` with JSON object | `{plan,deployment,deployment_version,runtime_directory_revision,runtime_directory_hash,runtime_routes}` |
| `GET` | `/overhead` | Get average scheduler overhead across agents. | None | Number of seconds |
| `POST` | `/scenario` | Update scheduler state with a processed task scenario. | Form field `data` with serialized task | `{accepted: boolean}` |
| `POST` | `/resource` | Update scheduler resource table for one device. | Form field `data` with JSON `{"device","resource"}` | `null` |
| `GET` | `/resource` | Get the full scheduler resource table. | None | Object keyed by device |
| `GET` | `/resource_lock` | Acquire resource ownership for a monitor probe such as bandwidth. | Form field `data` with JSON `{"resource","device"}` | `{holder}` |
| `GET` | `/source_nodes_selection` | Generate a source-to-edge-node selection plan within Backend-authorized candidates. | Form field `data` with JSON array containing `node_set`, `source_candidate_nodes`, and `source_selection_scope` | `{plan}` |
| `GET` | `/initial_deployment` | Generate initial deployment plan. | Form field `data` with JSON array | `{plan: {service: [node, ...]}}` |
| `GET` | `/redeployment` | Generate redeployment plan. | Form field `data` with JSON array | `{plan: {service: [node, ...]}}` |
| `GET` | `/generation_admission` | Decide whether one source may generate the next task. | Form field `data` with source request JSON | Policy-specific admission response |

Initial-deployment and redeployment policies have one canonical result contract: each key is a logical service name and
its value is a JSON list of target node names. Scheduler unions node lists for the same logical service across sources.
Node-to-services maps and scalar node values are invalid; Backend does not guess their orientation or repair an
incomplete plan. Scheduler and policy plugins enforce this shared contract through
`dependency/core/lib/scheduling/deployment_plan.py`. After validation, Backend may add the exact cloud node to every
service when `default-cloud-processor-backup` is enabled; this changes the published desired deployment, not the
Scheduler API response contract.

Source selection has a separate contract in `dependency/core/lib/scheduling/source_selection.py`. `node_set` contains
processor candidates; `source_candidate_nodes` contains the exact generator permission set resolved and persisted by
Backend. `selected_edge_nodes` selects from the former and `all_edge_nodes` selects from the latter. Scheduler never
queries Kubernetes or `RuntimeContext` to expand either set, and Backend independently rejects a returned source that
is not in the persisted source permission list.

### RuntimeDirectory control API

These endpoints are internal control-plane contracts. Backend is the only publisher; runtime readers consume
committed snapshots and task routes.

| Method | Path | Purpose | Request | Response |
| --- | --- | --- | --- | --- |
| `GET` | `/runtime-directory` | Read the canonical committed snapshot. | None | `{install_id,revision,directory_revision,nodes,deployment,routes,hash}` |
| `PUT` | `/runtime-directory` | Publish the initial snapshot with CAS. | Form `data` containing `{directory,expected_revision}` | Canonical committed snapshot |
| `DELETE` | `/runtime-directory` | Atomically clear the exact install's active snapshot and all indexed pending proposals. It does not wait for task leases. | Form `data` containing `{install_id}` | `{cleared,install_id,previous_revision}` |
| `POST` | `/runtime-directory/proposals` | Persist a candidate next revision. | Form `data` containing `{directory,base_revision,proposal_id,ttl_seconds}` | Proposal record |
| `POST` | `/runtime-directory/proposals/{proposal_id}/commit` | Atomically commit a persisted proposal and bound retirement of its base revision. | Form `data` containing `{expected_revision,retirement_grace_seconds}` | Canonical committed snapshot plus `retirement={revision,count,deadline,retired,revoked_count}` |
| `POST` | `/runtime-directory/proposals/{proposal_id}/reject` | Reject an uncommitted proposal. | Optional form `data` containing `{reason}` | Rejection record |

The directory validates positive revisions, unique logical slots/runtime ids, canonical node/deployment summaries, and
its content hash. Processor routes require a logical service and a complete endpoint tuple. Publication conflicts
return `409`; missing proposal/state returns `404`; invalid content returns `422`. In managed mode, the active snapshot
and unexpired proposals are Redis-backed and survive Scheduler Pod replacement. A production Scheduler without the
bootstrap Redis endpoint fails at startup; an in-memory store is available only through explicit test/local-harness
injection.

`DELETE /runtime-directory` is install-id guarded: clearing another install returns `409`, while clearing already
absent directory/proposal state succeeds with `previous_revision: 0`. One Redis script reads the install-scoped
proposal index and deletes the active snapshot, every indexed pending proposal, and the index atomically. Task-lease
keys are intentionally separate and retain their lease-derived expiry.

### Task lease API

| Method | Path | Purpose | Request | Response |
| --- | --- | --- | --- | --- |
| `GET` | `/runtime-directory/task-leases?revision=N` | Read lease and retirement status for one revision. | Query `revision` | `{revision,count,deadline,retired,revoked_count}` |
| `POST` | `/runtime-directory/task-leases` | Acquire a lease only for the currently active revision. | Form `data` with `{revision,root_uuid,ttl_seconds}` | `{revision,root_uuid,expires_at,valid_for_seconds}` |
| `PUT` | `/runtime-directory/task-leases` | Renew an existing unexpired lease. | Form `data` with `{revision,root_uuid,ttl_seconds}` | `{revision,root_uuid,expires_at,valid_for_seconds}` |
| `DELETE` | `/runtime-directory/task-leases` | Release an existing lease. | Form `data` with `{revision,root_uuid}` | `{revision,root_uuid,expires_at,released}` |
| `PATCH` | `/runtime-directory/task-leases` | Establish or reconcile one revision's immutable retirement deadline. The effective deadline may only stay the same or move earlier. | Form `data` with `{revision,deadline}` | `{revision,count,deadline,retired,revoked_count}` |

An acquire or renewal rejected by retirement returns
`{revision,root_uuid,retired:true,deadline}` instead of an expiry; runtime clients treat it as a hard stop. Releasing a
lease after its revision has retired is idempotent and reports it as already released.
`expires_at` is the Scheduler's wall-clock timestamp and is retained for observability. Runtime clients validate
`valid_for_seconds` and advance a local monotonic deadline from that relative duration, so lease safety does not
depend on wall-clock synchronization between cloud and edge nodes.
An inactive revision without a persisted retirement marker cannot renew: Scheduler rejects it fail-closed instead of
assuming that a missing marker means an unbounded grace period.

Production Scheduler stores directories/proposals and leases in Redis, scoped by install id and revision. Redis uses
a host-mounted AOF with synchronous fsync. Redeployment's proposal commit is one Scheduler transaction: it changes the
active directory from N to N+1, creates N's immutable retirement marker from Scheduler time, and clamps every N lease
to that deadline before exposing the result. Backend persists the returned authoritative deadline and normal
reconciliation reads status with `GET`; `PATCH` is reserved for uninstall's immediate fence. At the deadline Scheduler
atomically revokes the remainder and marks the revision retired. Exact-UID `Background` DELETE failures move into the
session's retryable cleanup backlog; an accepted request means the apiserver accepted the UID-guarded deletion or that
the exact UID is already absent, not that every dependent object has physically disappeared. The cleanup lane advances
independently from retirement and does not hold the next rollout gate. Normal uninstall sends `deadline=now` fences
without waiting for lease release, attempts the install-scoped directory/proposal clear, and proceeds with exact-UID
teardown even when Scheduler is unavailable. Scheduler is then deleted before the remaining workers as the definitive
admission fence. Every RuntimeService name includes both an installation digest and its revision, so a new installation
cannot reuse a name while an old installation's dependents are still being garbage-collected.
Expired lease entries are pruned on access. In-memory storage is reserved for explicit test/local-harness injection;
it is never an automatic production fallback.

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
The schedule is not usable unless Scheduler can also return an active positive directory revision and unambiguous
exact routes required by the plan; otherwise `/schedule` returns `503`.

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

Distributor releases the task lease only after the result is durably stored and Scheduler returns
`{"accepted": true}` for the scenario update. A release transport failure is logged and left to TTL expiry so a
normal task remains fail-safe; a redeploy retirement deadline still places a hard upper bound on old-revision
ownership.

Implementation note:

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
| Generator | Waits for schedulable exact routes, acquires the revision lease, copies routes into Task, and starts its run loop. | `dependency/core/generator/generator_server.py` |
| Monitor | Periodically samples `MON_PRAM` hooks and posts resource data to scheduler. | `dependency/core/monitor/monitor.py` |
| Datasource supervisor | Polls backend `/datasource_state` and starts or stops local source processes. | `datasource/datasource_server.py` |
| RTSP stream source | Reads `rtsp_video/manifest.json` through `VideoDataset` and streams clips to the configured RTSP address. | `datasource/rtsp_video.py` |
| Video dataset loader | Loads manifest-driven clip order, `video_root`, and frame-index metadata for both `http_video` and `rtsp_video`. | `datasource/video_dataset.py` |
| V4L2 getter | Opens a local camera device from `source.url` and creates generator tasks directly. | `dependency/core/lib/algorithms/data_getter/v4l2_video_getter.py` |
