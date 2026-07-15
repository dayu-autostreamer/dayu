# Operations Guide

This is the code-facing reference for Dayu lifecycle operations after Kubernetes/KubeEdge, the Dayu Sedna fork, and
the Dayu EdgeMesh fork are installed. Public first-run tutorials remain on the
[project documentation site](https://dayu-autostreamer.github.io/docs/).

## Non-Negotiable Prerequisites

The minimum released application runtime requires
[dayu-sedna `v1.1`](https://github.com/dayu-autostreamer/dayu-sedna/tree/v1.1) and
[dayu-edgemesh `v1.1`](https://github.com/dayu-autostreamer/dayu-edgemesh/tree/v1.1). Install the two tagged versions as a
matched pair; do not mix their legacy `v1.0` baselines with `v1.1`.

The released `v1.1` GM injects the authoritative `dayu.io/*` identities consumed by Dayu and EdgeMesh, so it remains
functionally compatible. The structural correction that also preserves caller-supplied
`spec.podTemplate.metadata.labels` and `annotations` is still in dayu-sedna's `Unreleased` source, not its historical
`v1.1` tag. When those custom fields are required, build a source revision containing the correction, apply its
RuntimeService CRD before restarting GM, and publish that revision under a new immutable maintenance tag. A controller
image update alone does not change the cluster's stored CRD schema. The schema upgrade prevents future pruning but
cannot restore metadata already removed from existing objects; recreate them with the normal Dayu uninstall/install
lifecycle (or a new runtime revision).

1. Install `runtimeservices.sedna.io` and the matching Sedna RBAC.
2. Run Sedna GM and every target-node LC from the same managed-runtime source revision. Installing only the CRD with
   upstream/old images is insufficient.
3. Run an EdgeMesh agent from the matching managed-runtime implementation on every target node. EdgeProxy must be
   enabled with `serviceFilterMode: FilterIfLabelExists`,
   `modules.edgeProxy.managedRuntime.enable: true`, and an explicit same-revision agent image.
4. On every cloud or edge target, verify `curl -fsS http://127.0.0.1:10551/readyz`. The API is loopback-only and is
   consumed by Sedna LC; it does not need a Kubernetes Service.
5. Ensure every selected Kubernetes node is `Ready` and has one Ready Sedna LC Pod and one Ready EdgeMesh agent Pod.

Backend preflight uses the exact node inventory plus selectors `sedna=lc` and
`k8s-app=kubeedge,kubeedge=edgemesh-agent`; missing/unready coverage fails install. If the cloud node carries
control-plane taints, the Sedna LC and EdgeMesh DaemonSet profiles must include matching tolerations so those agents
actually run there. Validate the prerequisite on each target rather than assuming an edge-only DaemonSet covers the
cloud runtime.

## Two Deployment Layers

| Layer | Started by | Resource model |
| --- | --- | --- |
| Platform support | [`dayu.sh`](../../dayu.sh) | namespace, backend-only RBAC, and support `JointMultiEdgeService` objects for persistent-AOF Redis, backend, frontend, and optional datasource |
| Application runtime | backend `POST /install` | one immutable Sedna `RuntimeService` per generator/scheduler/controller/processor/distributor/monitor slot and revision |

Support JMES Pods other than backend disable token mounting. Application RuntimeService Pods always disable token
mounting and Service-link environment injection. Backend runs as one process with the only Dayu ServiceAccount that
can call Kubernetes.

## Backend RBAC

`dayu.sh` creates a namespace-local Role for lifecycle writes and a namespace-suffixed,
read-only ClusterRole for cluster inventory. The suffix prevents one Dayu namespace from
overwriting or deleting another namespace's binding.

| API | Resources | Verbs | Purpose |
| --- | --- | --- | --- |
| namespaced `sedna.io` | `runtimeservices` | `get,list,watch,create,delete` | immutable lifecycle and condition watch |
| cluster core | `nodes` | `get,list,watch` | backend-owned inventory snapshot |
| cluster core | `pods` | `get,list,watch` | managed-agent preflight and exact-UID telemetry join |
| namespaced core | `configmaps` | `get,create,update,delete` | CAS runtime-session record |
| namespaced `metrics.k8s.io` | `pods` | `get,list` | optional telemetry snapshot |

Runtime workers need none of these permissions. Do not bind the backend role to a runtime Pod and do not add a
service-account token to a RuntimeService template.

## Start The Support Layer

`dayu.sh` expands repository-local `!include` values from `TEMPLATE/base.yaml` and consumes:

| Field | Effect |
| --- | --- |
| `namespace` | Dayu namespace; Kubernetes/Sedna/KubeEdge system namespaces are rejected. |
| `backend-rbac.*` | Backend ServiceAccount plus namespace-local manager and namespace-suffixed observer RBAC names. |
| `support-crd-meta.*` | API version/kind for support-layer JMES resources only. |
| `default-image-meta.*` | Registry/repository/tag for support containers. |
| `default-file-mount-prefix` | Host-path prefix used by rendered mounts. |
| `datasource.*` | Optional simulated datasource placement and data root. |
| `runtime.*` | activation, operation, inventory, telemetry, one bounded retirement grace, and task-lease budgets. |

```bash
TEMPLATE=template ACTION=start bash dayu.sh
```

This makes the UI/backend available; it does not install an application runtime.
Redis writes an AOF with synchronous fsync under
`<default-file-mount-prefix>/runtime-state/<namespace>/redis` on the cloud node; keep that host path writable and on
durable local storage to preserve Scheduler directory and lease recovery across Redis Pod replacement.

## Install Transaction

`POST /install` accepts a datasource label, policy id, and per-source DAG/node selection. Backend then:

1. normalizes catalog, DAG, source, processor-node inputs, and scheduler source-selection scope without contacting Kubernetes;
2. obtains one backend-owned Node snapshot plus one Sedna LC and one EdgeMesh agent Pod snapshot; required processor/cloud targets must be covered, while `all_edge_nodes` source permissions contain only Ready, jointly covered edge nodes;
3. creates a new install id/session and renders the scheduler RuntimeService at revision 1;
4. waits for scheduler `Activated=True` and `Ready=True`, including observed spec hash and exact object identities;
5. calls scheduler source-selection with the persisted `source_candidate_nodes` permission set, validates the result against that set, and independently validates initial processor placement against `node_set` plus the explicit cloud identity;
6. renders, creates, and activates the remaining RuntimeServices;
7. publishes RuntimeDirectory revision 1 to Scheduler's Redis-backed CAS store and reads it back to verify revision and canonical hash;
8. commits the session as `active` in ConfigMap `dayu-runtime-session` using `resourceVersion` compare-and-swap.

No partial runtime becomes routable. A missing plan, unknown target, ambiguous cloud node, failed preflight, rejected
spec, incomplete UID tuple, activation timeout, or directory readback mismatch fails the transaction and records the
error in the session.

## Runtime Routing And Leases

Runtime Pods consume `DAYU_RUNTIME_BOOTSTRAP`, which contains immutable install context, compact node metadata, support
endpoints, the active directory revision, and the lease TTL. It is not a Kubernetes cache and has no refresh operation.

Scheduler returns `runtime_directory_revision` and compact `runtime_routes` with each schedule. Generator copies them
into the Task and acquires `(revision, root_uuid)` once before first submission. Controller and Processor do not access
the lease API. Distributor performs the final renewal immediately before durable storage, requires Scheduler scenario
acknowledgement, and then releases. A transient acquire outage drops only the current task while Generator keeps its
source loop alive; a retired response requests a fresh schedule for the next round. The normal TTL covers end-to-end
processing. A failed release expires by TTL during normal operation; during redeployment the previous revision's
immutable retirement deadline clamps the lease and remains the hard upper bound.

Scheduler runs as one direct Uvicorn process. Its synchronous Redis-backed directory and lease endpoints execute in
FastAPI's thread pool, so Redis latency does not block the ASGI event loop or trigger a supervisor heartbeat timeout.

## Processor Rollout

`BackendCore.run_runtime_reconcile()` requests the current redeployment plan. Backend renders every desired processor slot
and compares its stable rollout hash with the active unit. The desired set is the complete validated Scheduler plan,
optionally unioned with one exact cloud slot per logical service when `default-cloud-processor-backup` is enabled.
This same composition is used during initial install. Placement, image, template, mount, or effective environment
changes cause a new RuntimeService revision; an unchanged cloud backup is retained when only an edge target changes.

The transaction is:

1. create and activate all changed/new processor RuntimeServices;
2. build the next complete RuntimeDirectory and persist the old revision's exact RuntimeService ownership with its
   deadline deliberately unarmed, closing the Backend crash boundary before publication;
3. propose the directory against the current base revision;
4. commit once in Scheduler: the same atomic boundary changes N to N+1, creates N's immutable deadline from
   `runtime.retirement-grace-seconds`, and clamps all existing N leases to that deadline;
5. verify readback when needed, persist Scheduler's authoritative retirement status, finalize the active session, and
   return immediately;
6. let the unified runtime-reconcile worker observe status and reclaim old ownership in bounded, retryable ticks.

Route publication and the retirement bound are one Scheduler transaction, so new tasks use the new routes while old
tasks retain their immutable old routes only during the bounded grace; there is no route-switch-to-fence gap. If
reclamation fails after commit, the new directory remains active and retired units remain owned by the session. At most one lease-protected retirement may be
pending. Another policy rollout is reported as deferred immediately only until that retirement finishes or reaches its
immutable deadline; it never waits behind an unbounded lease or Kubernetes finalizer.

Each `RuntimeOrchestrator.reconcile_retirement()` call performs one bounded step and never polls or sleeps internally.
Normal reconciliation reads Scheduler status with `GET`; it does not create or extend the deadline. Once the lease
count reaches zero, Backend submits UID-guarded `Background` deletion for the retired RuntimeServices and clears the
retirement record after every request is accepted or its exact UID is already absent. If the deadline arrives first,
Scheduler atomically revokes the remaining leases, records the forced count, and rejects further renewal before
Backend submits the same deletion. A rejected or timed-out delete moves those immutable identities to the persisted
`cleanup` backlog and releases the retirement gate. Every worker tick advances retirement and then independently
advances cleanup from the latest session snapshot, so continuous rollout cannot starve the backlog. Neither physical
garbage collection nor backlog failure blocks a subsequent rollout. There is no configurable quiet window and no
requirement that the retirement grace exceed the normal task-lease TTL.

`BackendCore.run_runtime_reconcile()` is the single worker for both retirement reconciliation and automatic policy
checks. It executes one reconciliation tick on its fixed internal cadence even when automatic redeployment is
disabled. `REDEPLOYMENT_REQUEST_INTERVAL` controls only when that worker asks the policy for another plan, and it waits
one interval before the first policy check because install has already committed the initial plan. Its lifecycle lock
protects only the worker token and is never held across Scheduler or Kubernetes I/O, so uninstall can invalidate the
worker promptly. RuntimeOrchestrator still serializes the one transaction or reconciliation step already in flight.
An automatic rollout propagates that cancellation token through activation, watch, publication, and exact-UID
retirement deletion; Scheduler calls have the separate `runtime.scheduler-request-timeout-seconds` cap.
Cancellation prevents further HTTP retries and interrupts retry backoff; an already-running synchronous request is
the only bounded cancellation delay. RuntimeService watches use a short server window with a slightly larger client
read timeout, so normal watch expiry is not misclassified as a transport failure.

Explicit install uses the same lifecycle cancellation contract. Backend
registers the install token before entering the serialized transaction, and a
stop request signals that token before waiting for cleanup. An overlapping stop
returns the same accepted state while the first request establishes the durable
uninstall intent; cancellation
is propagated through RuntimeService activation watches, Scheduler placement,
and initial directory publication. The last CAS session boundary is preserved
without recording a normal install failure, so stop can delete exact owned
RuntimeServices. Before the first directory publication, cleanup is Kubernetes
only and does not wait for an unready Scheduler endpoint. Cancellation latency
is therefore bounded by the current synchronous Kubernetes/HTTP request or the
short RuntimeService watch cancellation window, not the 300/900-second
operation timeout.

After the new directory CAS is finalized, management reads consume the immutable
session snapshot without taking the lifecycle transaction lock. Node inventory
uses a separate cache lock. Consequently, pending retirement is reconciled in
the background without making service-list/detail reads wait for old tasks or
the retirement deadline.

Runtime telemetry follows a stale-while-revalidate path owned by one backend daemon. It samples Scheduler `/resource`
and `/overhead` every `runtime.telemetry-sample-interval-seconds`; within that same non-overlapping worker, a slower
`runtime.metrics-sample-interval-seconds` due cycle batches every exact processor Pod reference in the committed
directory into one Pod list and one optional metrics list. Scheduler and Kubernetes calls have separate explicit
request budgets. `/system_parameters` and `/service_info/{service}` only deep-copy the last-known-good cache, so a slow
Scheduler, Metrics Server, kube-apiserver, or cluster DNS path cannot queue management API requests. Rebind/uninstall
invalidates the generation before any old in-flight result can publish.
Pod CPU/memory is summed across all expected containers and normalized against cached Node allocatable resources, with
an explicitly labeled capacity fallback. A Metrics API or whole Kubernetes sample failure retains each prior value as
`stale`; a missing/partial first sample is `unavailable`, never zero. Scheduler resource freshness is tracked separately
from scheduling overhead, so an overhead success cannot make an old bandwidth value appear fresh.

Available bandwidth is one edge-to-cloud WAN observation by design. The Scheduler lock permits one edge iperf client;
the cloud server and non-holder nodes report `-1`, and probe failures report `0`. Backend accepts exactly one finite
positive value, shares it across every service-detail row with the probe node identified, and fails closed when multiple
positive values violate the lock invariant. It must not be interpreted as a separate measurement for every edge node.
Frontend state and selected-service polling is settle-then-schedule (single-flight), aborts active fetches when polling
stops, and reads only this backend cache every five seconds; it does not change the Kubernetes sampling cadence.
Lifecycle and telemetry calls pass explicit bounded timeout budgets; bulk task and video transfers retain their
data-plane-specific timeout behavior instead of inheriting a control-plane deadline.

## Query Lifecycle

`POST /submit_query` opens one datasource label and is idempotent for that same label. A different label cannot open
until `POST /stop_query` closes the current one. Stopping a query clears in-memory task result queues and source-specific
visualization configuration; it does not uninstall RuntimeServices.

Query admission is enabled only after an active RuntimeDirectory is committed and is disabled atomically before
uninstall starts. Opening a query has no source-count-dependent sleep. One generation-scoped collector polls the
captured Distributor endpoint with `runtime.result-request-timeout-seconds` and requests at most
`runtime.result-batch-size` records; cancellation invalidates that generation, so a late response cannot populate a
new query or restore lifecycle-owned endpoint state.

When `datasource.use-simulation=false`, backend opens the selected datasource automatically after successful install.

## Graceful Uninstall And System Stop

Backend `POST /stop_service` is an asynchronous administrative command. The first accepted request closes query
admission, persists an `operation_id` with `phase="uninstalling"`, starts the lifecycle reconcile worker, and returns
once that intent is durable. Repeated or concurrent requests reuse the same intent and worker and return immediately.
The API does not wait for Kubernetes deletion. Clients must poll `GET /install_state` until it reports
`state="uninstall"` and `phase="uninstalled"`; while work remains it reports `uninstalling` or
`finalizing-uninstall`, and `last_error` exposes the latest retryable failure.

The background operation uses the CAS session as its exact ownership record:

1. delete all generator RuntimeServices by exact UID so no new task can enter the runtime;
2. send an immediate `deadline=now` retirement fence for every active, retiring, or plausibly publishing directory
   revision; Scheduler atomically revokes any remaining leases;
3. attempt the install-id-guarded atomic clear of the active RuntimeDirectory and indexed proposals;
4. persist `finalizing-uninstall`, then delete Scheduler RuntimeService(s) by exact UID;
5. delete controller/processor/distributor/monitor units by exact UID;
6. delete `dayu-runtime-session` after all owned exact-UID `Background` DELETE requests are accepted or their targets
   are already absent.

Uninstall does not poll lease counts, wait for releases, or inherit the redeployment grace. A Scheduler fence or
directory-clear failure is logged but cannot make task state veto administrative teardown. Scheduler remains available
through the best-effort fence and clear, then its own exact-UID deletion is the definitive admission fence: it is
removed before workers so no surviving control plane can route to workers already being torn down. Exact name/UID
ownership checks still prevent a retry from deleting a replacement object. Once `finalizing-uninstall` is persisted, a
retry needs no Scheduler API and resumes Scheduler/worker deletion before removing the ConfigMap CAS record. A failed
exact-UID API request preserves that retry boundary and error. Acceptance does not wait for Sedna-owned dependents or
finalizers to disappear physically; Kubernetes garbage collection proceeds asynchronously. RuntimeService names carry
an installation digest as well as the revision, preventing an immediate reinstall from colliding with dependents of
the old UID during that asynchronous cleanup. There is no local manifest fallback.

```bash
TEMPLATE=template ACTION=stop bash dayu.sh
```

The script gives backend `/stop_service` a bounded, best-effort opportunity to fence and remove the managed runtime
before deleting the support layer. It treats the POST response as acceptance and polls `/install_state` for session
removal within the same budget. `GRACEFUL_STOP_WAIT_SEC` may override the default 60-second budget. A
backend failure, timeout, malformed response, or unavailable service is logged but never blocks `ACTION=stop`: the
script continues by deleting RuntimeServices, support resources, Services/Endpoints, workloads, access bindings, and
the namespace. Per-kind deletion is asynchronous and does not repeat Service/Endpoint/Pod disappearance polling;
the bounded Namespace deletion is the single completion barrier before the command reports success. This keeps the
public stop interface independent of backend health, as in earlier Dayu versions.

`WAIT_EDGEMESH_RULES=false` may skip the best-effort EdgeMesh iptables cleanup wait. When shell cleanup follows an
unsuccessful graceful uninstall, Scheduler's install-id-guarded directory/proposal cleanup may not complete and the
host-mounted Redis directory remains intact. Its keys are install-id scoped and cannot route a later installation;
pending proposals and ordinary task-lease records retain their own expiry behavior. Remove persisted Redis data
manually only while the support Redis is stopped and discarding the old installation state is intentional.

## Why Edge Nodes Do Not Pay Kubernetes Discovery Cost

- the Kubernetes Python package exists in the backend image only;
- runtime Pods have no service-account token and no kubeconfig/API endpoint;
- routes are exact immutable task data, not Pod/Node/Service lookups;
- Scheduler's durable Redis directory/proposal store and task-lease store are application protocols, not Kubernetes caches;
- Hedger and the other Scheduler policies never delete or request deletion of processor Pods; failed workload reconciliation and exact RuntimeService retirement remain backend control-plane responsibilities;
- EdgeMesh projects its existing MetaServer-backed Service/Endpoints informers into an atomic in-memory exact route;
- backend telemetry periodically batches exact Pod UID joins for the entire active directory into one
  server-side-label-filtered Pod list and at most one equally filtered metrics list, while reusing its independent node
  inventory snapshot; browser service-detail requests only read this normalized last-known-good snapshot, including
  the single shared edge-to-cloud probe projection;
- automatic redeploy reuses the same Backend-owned node TTL and does not repeat Sedna/EdgeMesh Agent Pod lists after
  installation has authorized the immutable processor and source candidate sets.

There is therefore no normal runtime path that can issue Kubernetes calls from an edge worker, and no forced cache
refresh path to regress into repeated lists. Keep `tests/unit/core_lib/test_runtime_kubernetes_boundary.py` passing to
preserve this property.

## Useful Checks

```bash
kubectl get nodes
kubectl get pods -A -l sedna=lc -o wide
kubectl get pods -A -l 'k8s-app=kubeedge,kubeedge=edgemesh-agent' -o wide
kubectl get runtimeservices.sedna.io -n dayu
kubectl get runtimeservices.sedna.io -n dayu -o custom-columns='NAME:.metadata.name,REV:.spec.deploymentRevision,READY:.status.conditions[?(@.type=="Ready")].status,ACTIVATED:.status.conditions[?(@.type=="Activated")].status'
kubectl get cm -n dayu dayu-runtime-session -o yaml
```

On each target node:

```bash
curl -fsS http://127.0.0.1:10551/readyz
```

For a failed activation, inspect RuntimeService conditions, its exact Deployment/Pod/Service/Endpoints incarnation,
Sedna GM/LC logs, and the EdgeMesh exact-route status by Service UID. For source or visualization problems, inspect
backend `/query_state`, `/datasource_state`, datasource manifests, and the relevant visualization hook configuration;
those are separate from the managed-runtime publication path.
