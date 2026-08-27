# Runtime Service APIs

This document describes the internal APIs used by Dayu runtime components. These are repository-internal service contracts, not frontend-facing APIs.

## Shared Runtime Contracts

| Topic | Behavior |
| --- | --- |
| Task payload | Internal services exchange serialized `Task` strings produced by `Task.serialize()`. Root tasks carry `schedule_decision_id` and `schedule_plan_digest`; forks preserve them. Source replay timing remains private to the selected data-getter and is never serialized into a `Task`. |
| File transfer | Binary task content is sent as `multipart/form-data` with a `file` field plus a `data` field containing the serialized task. |
| Scheduler/resource updates | Scheduler endpoints often receive JSON encoded into a form field named `data`. |
| Runtime bootstrap | `DAYU_RUNTIME_BOOTSTRAP` supplies immutable install context and static infrastructure endpoints; it is never a Kubernetes discovery cache. |
| Exact task routing | `Task` serializes `runtime_directory_revision`, `runtime_routes`, and `root_uuid`. Controller/processor routes must be complete exact identities. |
| Task ownership and retirement | A task-aware schedule creates one expiring reservation keyed by `root_uuid`. Generator either materializes the Task and atomically promotes the reservation at lease admission, or cancels it when the data getter produces no Task. Active execution is keyed by `(runtime_directory_revision, root_uuid)`. Distributor performs final renew/persist followed by scenario/release completion. Backend may place one immutable deadline on an old revision. |
| Task delivery ACK | A task write succeeds only with `{accepted: true, task_uuid: "<exact task uuid>"}`. `200 null`, a mismatched UUID, timeout, or connection failure never transfers ownership. |
| Task artifact | All branches of one lease identity share one immutable, atomically published node-local artifact. Different roots/revisions cannot reuse the same temp path. |

Runtime service code does not import Kubernetes, load a kubeconfig, or discover Pods, Nodes, or Services. Missing or
ambiguous task routes fail closed.

## Controller Service

Implementation entrypoint: `dependency/core/controller/controller_server.py`

| Method | Path | Purpose | Request | Response |
| --- | --- | --- | --- | --- |
| `POST` | `/check` | Check processors referenced by exact local runtime routes. | Optional form `data` with JSON containing `runtime_routes` | `{status: "ok"|"not ok"}` |
| `POST` | `/submit_task` | Accept a new task from generator or another controller. | Multipart with `file` and serialized `data` | Exact task delivery ACK after downstream ownership transfer |
| `POST` | `/process_return_task` | Accept a processed task returned by processor. | Form field `data` with serialized task | Exact task delivery ACK after Redis retention or downstream ownership transfer |
| `POST` | `/processor_queues_clear` | Fan out a dry-run or destructive queue-clear request through exact local processor routes. | Form `data` with `{runtime_routes,services?,timeout_s?,reason?,max_count?,dry_run?}` | Aggregate `{ok,device,service_count,matched_count,cleared_count,remaining_count,services}` |

Operational notes:

- The FastAPI lifespan owns one task-lease-TTL `FileCleaner`. Controller startup, shutdown, intermediate DAG branches,
  and processor queue clearing never delete the shared artifact directory or a sibling branch's artifact.
- Uploads are atomically published with `os.replace`; active Controller/Processor transitions refresh the artifact
  timestamp. Files with no progress for one full task-lease TTL are reclaimed without another cleanup setting.
- `submit_task` returns only after the file is complete and every selected next hop has acknowledged ownership.
- Parallel joins use predecessor service names as idempotency keys. A ready join remains in Redis until its merged task
  is acknowledged downstream, so retrying one branch cannot consume or over-count the barrier. The same generic
  barrier store supports exact-key read-only snapshots without scanning the Redis namespace.
- Controller resolves downstream controller/processor targets only from the Task's exact route snapshot. It does not
  call the task-lease API; a missing or invalid route stops forwarding.

## Processor Service

Implementation entrypoint: `dependency/core/processor/processor_server.py`

| Method | Path | Purpose | Request | Response |
| --- | --- | --- | --- | --- |
| `GET` | `/health` | Basic health probe. | None | `{status: "ok"}` |
| `POST` | `/predict` | Queue a task that includes a file payload. | Multipart with `file` and serialized `data` | Exact task delivery ACK after atomic file publication and queue insertion |
| `POST` | `/predict_local` | Queue a task that uses the shared local artifact. | Form field `data` | Exact task delivery ACK after artifact validation and queue insertion |
| `POST` | `/predict_and_return` | Process a task synchronously and return the serialized result. | Multipart with `file` and serialized `data` | Serialized task string or `null` |
| `GET` | `/queue_state` | Return one atomic snapshot of the background processor lane. | None | `{waiting_count,waiting_tasks,busy,running_elapsed_s,running_phase,phase_elapsed_s,capacity,sequence,running_task,observed_at}` |
| `POST` | `/queue_clear` | Preview or remove queued processor tasks. | Form field `data` with JSON such as `{"dry_run": true, "max_count": 10, "reason": "manual_queue_clear"}` | Queue clear summary with matched, cleared, remaining, and dropped task metadata |
| `GET` | `/model_flops` | Return the processor model FLOPs value. | None | Numeric FLOPs value |
| `GET` | `/model_memory` | Return the processor process RSS in bytes. | None | Integer memory usage |

Operational notes:

- `PROCESSOR_NAME` selects the processor implementation.
- `PRO_QUEUE_NAME` selects the queue strategy.
- A background thread drains the task queue and posts results back to controller through `/process_return_task`.
  A non-dropping queue requeues processing failures/empty results, and a completed inference result stays in the worker
  until Controller returns the exact ACK; delivery retries do not run inference again.
- Queue insertion, dequeue-to-running ownership, failure requeue, completion, and queue clearing update the queue state
  under the same lock. Built-in queues therefore expose ordered `waiting_tasks`; a custom queue that cannot provide a
  non-destructive ordered snapshot reports that field as `null`. `running_phase` is `processing` or `handoff`, and the
  lane remains busy during handoff exactly as it did while retaining a result for Controller acknowledgement.
- Fork UUIDs are deterministic for one root-task DAG stage. Processor remembers accepted UUIDs for one lease
  TTL and ACKs a replay without enqueuing it again, giving the HTTP path at-least-once delivery without duplicate inference.
- Processor validates the exact task route and does not call the task-lease API or resolve service placement from
  local or cluster state. Normal end-to-end execution is covered by the lease TTL acquired by Generator.
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

Scheduler runs as one direct Uvicorn process, preserving its single-process scheduling-agent state without a second
Gunicorn worker supervisor. Redis-backed RuntimeDirectory and task-lease handlers are synchronous FastAPI endpoints,
so blocking Redis work runs in the framework thread pool instead of starving the ASGI event loop. Removing the second
supervisor also removes its worker-heartbeat timeout from this path.

| Method | Path | Purpose | Request | Response |
| --- | --- | --- | --- | --- |
| `GET` | `/schedule` | Generate a schedule plan plus exact routes for one source. | Form field `data` with the current source, DAG, deployment/directory cache state, and optional `task_context` | Always returns `{plan,deployment_version,runtime_directory_revision,runtime_directory_hash,runtime_directory_unchanged,runtime_routes_cache_key,runtime_routes_cached,schedule_decision}`; `deployment` and `runtime_routes` are included only when the caller's caches are not current |
| `GET` | `/overhead` | Get average scheduler overhead across agents. | None | Number of seconds |
| `POST` | `/scenario` | Update scheduler state with a processed task scenario. | Form field `data` with serialized task | `{accepted: boolean}` |
| `POST` | `/resource` | Update scheduler resource table for one device. | Form field `data` with `{device,resource,runtime_directory_revision?}` | `null` |
| `GET` | `/resource` | Get the full scheduler resource table. | None | Object keyed by device |
| `GET` | `/resource_lock` | Acquire resource ownership for a monitor probe such as bandwidth. | Form field `data` with JSON `{"resource","device"}` | `{holder}` |
| `GET` | `/source_nodes_selection` | Generate a source-to-edge-node selection plan within Backend-authorized candidates. | Form field `data` with JSON array containing `node_set`, `source_candidate_nodes`, and `source_selection_scope` | `{plan}` |
| `GET` | `/initial_deployment` | Generate initial deployment plan. | Form field `data` with JSON array | `{plan: {service: [node, ...]}}` |
| `GET` | `/redeployment` | Generate redeployment plan. | Form field `data` with JSON array | `{plan: {service: [node, ...]}}` |
| `GET` | `/generation_admission` | Decide whether one source may generate the next task. | Form field `data` with source request JSON | Policy-specific admission response |

Expected Scheduler rejections use FastAPI's structured `{"detail": ...}` response. Backend management calls preserve
a bounded, single-line form of the actionable detail together with the HTTP status and endpoint; validation inputs are
omitted before an install or redeployment failure is recorded in the existing `RuntimeSession.last_error` field and
exposed by `/install_state` without adding another lifecycle field.
The Scheduler also writes one bounded, single-line warning for each 4xx rejection so the same failure is actionable
from either the frontend or the Scheduler log. Request bodies and complete DAG/policy payloads are not logged.

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

The `/schedule` response is cache-aware. A matching `(runtime_directory_revision, runtime_directory_hash)` suppresses
the unchanged `deployment`; when `runtime_route_cache_keys` also contains the returned
`runtime_routes_cache_key`, the matching `runtime_routes` payload is omitted. Generator retains only host-returned
cache state and fails closed if it cannot reconstruct the effective directory and exact routes. Resource telemetry
records its reported directory revision; a scheduling extension may treat a sample as current LIVE state only when
that revision matches the snapshot revision.

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

### Task lease and reservation API

| Method | Path | Purpose | Request | Response |
| --- | --- | --- | --- | --- |
| `GET` | `/runtime-directory/task-leases?revision=N` | Read lease and retirement status for one revision. | Query `revision` | `{revision,count,deadline,retired,revoked_count}` |
| `POST` | `/runtime-directory/task-leases` | Acquire a lease only for the currently active revision and register its optional immutable scheduling commitment. | Form `data` with `{revision,root_uuid,ttl_seconds,commitment?}` | `{revision,root_uuid,expires_at,valid_for_seconds}` |
| `PUT` | `/runtime-directory/task-leases` | Renew an existing unexpired lease. | Form `data` with `{revision,root_uuid,ttl_seconds}` | `{revision,root_uuid,expires_at,valid_for_seconds}` |
| `DELETE` | `/runtime-directory/task-leases` | Release an existing lease. | Form `data` with `{revision,root_uuid}` | `{revision,root_uuid,expires_at,released}` |
| `PATCH` | `/runtime-directory/task-leases` | Establish or reconcile one revision's immutable retirement deadline. The effective deadline may only stay the same or move earlier. | Form `data` with `{revision,deadline}` | `{revision,count,deadline,retired,revoked_count}` |
| `DELETE` | `/runtime-directory/task-reservations` | Cancel a task-bound scheduling decision when no Task was materialized. | Form `data` with `{revision,root_uuid,decision_id?}` | `{revision,root_uuid,cancelled,already_cancelled?,decision_id}` |

An acquire or renewal rejected by retirement returns
`{revision,root_uuid,retired:true,deadline}` instead of an expiry; runtime clients treat it as a hard stop. Releasing a
lease after its revision has retired is idempotent and reports it as already released.
`expires_at` is the Scheduler's wall-clock timestamp and is retained for observability. Runtime clients validate the
Scheduler-computed relative `valid_for_seconds` instead of comparing cloud and edge wall clocks.
An inactive revision without a persisted retirement marker cannot renew: Scheduler rejects it fail-closed instead of
assuming that a missing marker means an unbounded grace period.

The optional acquire-time `commitment` contains the admitted root's source/task identity, creation timestamp,
decision id and plan digest, deployment version, runtime-directory revision, DAG device mapping, and task metadata.
Scheduler validates that its root and revision match the lease. For a task-aware `/schedule` request, Scheduler first persists an expiring
`pending` record containing the returned decision and plan. Lease acquisition checks the immutable decision fields and
atomically promotes that record to `active`; expiry, release, and retirement remove it with the lease. Requests without
`task_context`, and later tasks that intentionally reuse an earlier decision, retain their existing behavior and create
the active record directly at lease admission. Repeating a task-aware schedule request replays the same pending
decision without invoking the agent again; an old-revision pending record is replaced only after the active directory
has moved to a new revision.

`Scheduler.get_scheduling_snapshot(scope)` is an in-process `SCH_AGENT` contract, not an HTTP endpoint or a second
route-authority store. It returns a mutation-safe snapshot at one of two explicit scopes:

| Scope | Runtime state included | Intended use |
| --- | --- | --- |
| `COMMITTED` (default) | Current deployment and telemetry, including `resource_received_at` and `resource_runtime_revision`, plus current `reservations`, active `commitments`, and exact active `task_barriers` | Commitment-aware and future-state scheduling |
| `LIVE` | The same deployment and telemetry fields, with empty reservations, commitments, and barriers | Immediate decisions over executable placement and current telemetry |

Extensions should request the smallest scope they need through `SchedulingSnapshotScope`. Before consuming a resource
sample as LIVE state, they must verify that its `resource_runtime_revision` equals the snapshot's
`runtime_directory_revision`. Commitment records and barriers are Redis-backed in production and survive Scheduler
replacement while their leases remain valid.

The normal per-task sequence is deliberately small: a task-aware schedule reserves once; Generator either acquires
once before first submission or cancels the exact reservation when no Task materializes. Controller and Processor
never access the lease API. Distributor renews once immediately before durable persistence, sends the scenario update,
and releases only after an explicit acknowledgement. The configured TTL covers both the short pending reservation and
normal end-to-end processing.
When a redeployment retires that task's old revision, the immutable retirement deadline clamps the lease and is the
hard upper bound regardless of the original TTL.

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
`Foreground` teardown even when Scheduler is unavailable. Scheduler is then deleted before the remaining workers as the
definitive admission fence. Unlike the rollout cleanup lane, uninstall retains its RuntimeSession until a complete
resource snapshot proves that every RuntimeService and its Deployment, ReplicaSet, Pod, Service, Endpoints, and
EndpointSlice dependents are absent. Every RuntimeService name includes both an installation digest and its revision,
but name separation is not used as permission to overlap old and new install generations.
Expired lease entries are pruned on access. In-memory storage is reserved for explicit test/local-harness injection;
it is never an automatic production fallback.

`/schedule` expects data close to the internal serialized Task-DAG shape, including synthetic `_start` and `_end`
nodes:

```json
{
  "source_id": 0,
  "meta_data": {"resolution": "720p", "fps": 5, "buffer_size": 4, "encoding": "mp4v"},
  "current_configuration": {"resolution": "720p", "fps": 5, "buffer_size": 4, "encoding": "mp4v"},
  "source_device": "edgex1",
  "all_edge_devices": ["edgex1", "edgex2"],
  "deployment_version": 3,
  "runtime_directory_revision": 7,
  "runtime_directory_hash": "sha256-directory-hash",
  "runtime_route_cache_keys": ["sha256-route-cache-key"],
  "dag": {
    "_start": {
      "service": {"service_name": "_start", "execute_device": "edgex1"},
      "prev_nodes": [],
      "next_nodes": ["car-detection"]
    },
    "car-detection": {
      "service": {"service_name": "car-detection", "execute_device": "cloudx1"},
      "prev_nodes": ["_start"],
      "next_nodes": ["_end"]
    },
    "_end": {
      "service": {"service_name": "_end", "execute_device": "cloudx1"},
      "prev_nodes": ["car-detection"],
      "next_nodes": []
    }
  },
  "task_context": {
    "source_id": 0,
    "task_id": 42,
    "task_uuid": "root-task-uuid",
    "root_uuid": "root-task-uuid"
  },
  "schedule_request_attempt": 1
}
```

Different `GEN_BSO` implementations may append scheduler-specific observations such as `skip_count`, `frame`, or
`hash_code`. Generator owns and overwrites the source identity, raw/current configuration, current DAG, candidate
devices, deployment version, runtime-directory state, and `task_context`; hooks cannot fabricate or freeze these host
fields. A per-task scheduling agent may return any partial combination of configuration fields, `dag` offloading, and
`deployment_version`. Missing fields preserve current state, while an absent/unroutable first DAG is completed by the
selected startup-policy hook. Replica placement remains controlled by the independent initial-deployment and
redeployment hook lifecycle; `/schedule` returns its effective deployment and version for task attribution.
`schedule_decision` binds the response plan to the reserved root through a stable decision id and canonical plan
digest. A positive scheduling interval may reuse that decision for later root tasks, but never reuses their task UUID.
The schedule is not usable unless Scheduler can also return an active positive directory revision and unambiguous
exact routes required by the plan; otherwise `/schedule` returns `503`.

## Distributor Service

Implementation entrypoint: `dependency/core/distributor/distributor_server.py`

| Method | Path | Purpose | Request | Response |
| --- | --- | --- | --- | --- |
| `POST` | `/distribute` | Persist a finished task and forward scenario data to scheduler. | Multipart with `file` and serialized `data` | Exact task delivery ACK after durable SQLite persistence |
| `GET` | `/result` | Incrementally fetch stored task results. | JSON body `{"size","time_ticket"}` | `{result, time_ticket, size}` |
| `GET` | `/file` | Download a generated file and schedule it for deletion. | JSON body `{"file":"<path>"}` | File response |
| `GET` | `/result_by_time` | Query results for a time range. | JSON body `{"start_time","end_time","source_id?"}` | `{result, size}` |
| `GET` | `/all_result` | Dump all stored results. | None | `{result, size}` |
| `GET` | `/export_result_log` | Export stored results as a gzip-compressed JSON file. | None | `application/gzip` file |
| `POST` | `/clear_database` | Clear the result database. | None | `null` |
| `GET` | `/is_database_empty` | Check whether the result database contains records. | None | Boolean |

Distributor first accepts an idempotent retry already stored for the same root identity, or renews the task lease to
reject a new result that arrived after its revision retired and then stores it durably. Durable persistence defines the
delivery ACK boundary. Scheduler scenario feedback and lease release are post-persistence completion work: failures are
reported but cannot turn an already stored result back into a lost task; an unreleased lease converges through its TTL
or immutable retirement deadline.

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

After a source is registered under `path=<source-path>`, four routes become available:

| Method | Path | Purpose | Request | Response |
| --- | --- | --- | --- | --- |
| `GET` | `/<source-path>/source` | Generate the next buffered clip for that source. | Form field `data` with JSON request | JSON array of frame hash or frame index values |
| `GET` | `/<source-path>/file` | Download the clip generated by the previous `/source` call. | None | File response |
| `GET` | `/<source-path>/shared_file` | Return the node-local temp artifact path for a colocated Generator. | None | `{"file_name":"<shared temp path>"}` |
| `GET` | `/<source-path>/status` | Report source-process identity and finite-stream state. | None | `{instance_id,exhausted,ready}` |

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

The service resolves those hook names dynamically, applies frame filtering, frame processing, and compression, then
publishes one temp artifact. A colocated Generator accepts the `/shared_file` path only when the file exists locally;
otherwise it downloads the same artifact through `/file`. `/status` lets the getter distinguish a recreated source
process from one that remains exhausted.

## V4L2 Video Getter

Implementation entrypoint: `dependency/core/lib/algorithms/data_getter/v4l2_video_getter.py`

`v4l2_video` is used for real-camera datasource configs such as `config/datasource_configs/real_camera.yaml`. It is a
generator data getter rather than a datasource-supervisor service, so it does not expose a repository-managed HTTP API
or use the `http_video`/`rtsp_video` manifest layout.

## Internal Non-HTTP Entry Points

These runtime components are important for understanding the system but do not expose repository-managed public HTTP routes:

| Component | Behavior | Main code path |
| --- | --- | --- |
| Generator | Filters an ingestion round, reserves a TaskIdentity, and schedules before materializing source data. It then either creates the Task and acquires its revision lease plus commitment, or cancels the exact scheduling reservation when the getter returns no Task. A transient lease outage retains and retries a materialized task; a retired revision requests a fresh schedule while the source loop remains alive. | `dependency/core/generator/generator_server.py` |
| Monitor | Periodically samples `MON_PRAM` hooks and posts resource data to scheduler. | `dependency/core/monitor/monitor.py` |
| Datasource supervisor | Polls backend `/datasource_state` and starts or stops local source processes. | `datasource/datasource_server.py` |
| RTSP stream source | Reads `rtsp_video/manifest.json` through `VideoDataset` and streams clips to the configured RTSP address. | `datasource/rtsp_video.py` |
| Video dataset loader | Loads manifest-driven clip order, `video_root`, and frame-index metadata for both `http_video` and `rtsp_video`. | `datasource/video_dataset.py` |
| V4L2 getter | Opens a local camera device from `source.url` and creates generator tasks directly. | `dependency/core/lib/algorithms/data_getter/v4l2_video_getter.py` |
