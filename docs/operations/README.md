# Operations Guide

This is the code-facing reference for Dayu lifecycle operations after Kubernetes/KubeEdge, the Dayu Sedna fork, and
the Dayu EdgeMesh fork are installed. Public first-run tutorials remain on the
[project documentation site](https://dayu-autostreamer.github.io/docs/).

## Non-Negotiable Prerequisites

The application runtime requires
[dayu-sedna `v1.1`](https://github.com/dayu-autostreamer/dayu-sedna/tree/v1.1) and
[dayu-edgemesh `v1.1`](https://github.com/dayu-autostreamer/dayu-edgemesh/tree/v1.1). Install the two tagged versions as a
matched pair; do not mix their legacy `v1.0` baselines with `v1.1`:

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
| `runtime.*` | activation, operation, inventory, drain, quiet-window, and lease budgets. |

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
into the Task and acquires `(revision, root_uuid)` before first submission. Controller renews at forwarding boundaries;
Processor also maintains a bounded heartbeat throughout long inference so one stage cannot silently outlive its TTL.
Distributor releases it only after durable storage and scheduler scenario acknowledgement. Any acquire/renew result
that is not explicitly acknowledged stops task progress; a failed release expires by TTL and conservatively delays
drain.

## Processor Rollout

`BackendCore.run_cycle_deploy()` requests the current redeployment plan. Backend renders every desired processor slot
and compares its stable rollout hash with the active unit. The desired set is the complete validated Scheduler plan,
optionally unioned with one exact cloud slot per logical service when `default-cloud-processor-backup` is enabled.
This same composition is used during initial install. Placement, image, template, mount, or effective environment
changes cause a new RuntimeService revision; an unchanged cloud backup is retained when only an edge target changes.

The transaction is:

1. create and activate all changed/new processor RuntimeServices;
2. build the next complete RuntimeDirectory;
3. propose it against the current base revision;
4. commit with compare-and-swap and verify readback;
5. wait until leases on the old revision remain zero for `runtime.drain-quiet-window-seconds`;
6. delete retired RuntimeServices by exact name and UID.

Publication precedes retirement, so new tasks use the new routes while old tasks retain their immutable old routes. If
drain/deletion fails after commit, the new directory remains active and retired units remain in the session for retry;
they are never silently forgotten or deleted before drain.

`REDEPLOYMENT_REQUEST_INTERVAL` controls the periodic policy check. It does not weaken activation or drain gates.

## Query Lifecycle

`POST /submit_query` opens one datasource label and is idempotent for that same label. A different label cannot open
until `POST /stop_query` closes the current one. Stopping a query clears in-memory task result queues and source-specific
visualization configuration; it does not uninstall RuntimeServices.

When `datasource.use-simulation=false`, backend opens the selected datasource automatically after successful install.

## Graceful Uninstall And System Stop

Backend `POST /stop_service` uses the CAS session as its exact ownership record:

1. delete all generator RuntimeServices first so no new task lease can be acquired;
2. drain every active or plausibly pending RuntimeDirectory revision;
3. persist `clearing-directory`, then atomically clear the install-scoped active RuntimeDirectory snapshot and all
   indexed pending proposals;
4. delete controller/processor/distributor/monitor units by exact RuntimeService UID only after the empty directory is
   durably observable;
5. persist `finalizing-uninstall` with only Scheduler identities;
6. delete Scheduler RuntimeService(s) last so directory and lease APIs remain available through drain and clear;
7. delete `dayu-runtime-session` only after all resource deletion succeeds.

Backend persists `clearing-directory` before step 4 and advances to `finalizing-uninstall` only after Scheduler
acknowledges the atomic directory/proposal clear or an empty revision-0 readback proves that an ambiguous request
committed. A retry in the former phase repeats the idempotent clear and finishes the still-recorded non-Scheduler UID
deletions; a retry in the latter phase never requires Scheduler and finishes Scheduler UID deletion plus ConfigMap CAS.
The directory is therefore never left advertising a route whose RuntimeService has already been deleted. Failed
uninstall preserves a precise retry boundary and error. There is no local manifest fallback.

```bash
TEMPLATE=template ACTION=stop bash dayu.sh
```

The script calls backend `/stop_service` before deleting the support layer. If RuntimeServices remain and graceful
drain fails or backend is unavailable, the script preserves the namespace and exits non-zero. Only use the destructive
override when abandoning in-flight tasks is intentional:

```bash
FORCE_RUNTIME_STOP=true TEMPLATE=template ACTION=stop bash dayu.sh
```

The override deletes RuntimeServices and support resources without a successful lease drain. `WAIT_EDGEMESH_RULES=false`
may skip the best-effort EdgeMesh iptables cleanup wait, but does not change runtime drain semantics. Forced cleanup
also bypasses Scheduler's install-id-guarded directory/proposal deletion and leaves the host-mounted Redis directory
intact. Because keys are install-id scoped, they cannot route a later install, but repeated forced stops can retain
unreachable active snapshots. Pending proposals and task leases still age out through their own TTLs, but the host data
is not erased; remove it only when the support Redis is stopped and abandonment is intentional.

## Why Edge Nodes Do Not Pay Kubernetes Discovery Cost

- the Kubernetes Python package exists in the backend image only;
- runtime Pods have no service-account token and no kubeconfig/API endpoint;
- routes are exact immutable task data, not Pod/Node/Service lookups;
- Scheduler's durable Redis directory/proposal store and task-lease store are application protocols, not Kubernetes caches;
- Hedger and the other Scheduler policies never delete or request deletion of processor Pods; failed workload reconciliation and exact RuntimeService retirement remain backend control-plane responsibilities;
- EdgeMesh projects its existing MetaServer-backed Service/Endpoints informers into an atomic in-memory exact route;
- backend telemetry batches exact Pod UID joins into one server-side-label-filtered Pod list and at most one equally
  filtered metrics list, while reusing its node inventory snapshot;
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
