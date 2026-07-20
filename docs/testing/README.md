# Testing Strategy

This document explains how Dayu's current code structure maps to the test pyramid and where hook-focused tests should live.

## Repository Structure

| Area | Responsibility | Main paths |
| --- | --- | --- |
| Control plane | Backend APIs, install/query orchestration, visualization config handling | `backend/` |
| Runtime services | Generator, scheduler, controller, processor, distributor, monitor | `dependency/core/` |
| Shared runtime library | Hook registry, context resolution, DAG/task types, scheduling contracts, network helpers, algorithms | `dependency/core/lib/` |
| Source adapters | `http_video`, `rtsp_video`, dataset readers, datasource API | `datasource/` |
| Service entrypoints | Container-facing `main.py` launchers for components | `components/` |
| Deployment templates | YAML-driven runtime composition and hook selection | `template/` |
| Build configuration | Dockerfiles, Bake targets, and template image references | `build/`, `docker-bake.hcl`, `tools/validate_build_matrix.py` |
| Tests | Layered Python test pyramid | `tests/` |

## Current Test Pyramid

The repository already follows a sensible layered layout that is close to mature Python infrastructure projects:

| Layer | Goal | Current path | CI job |
| --- | --- | --- | --- |
| Unit | Fast checks for pure logic and isolated runtime helpers | `tests/unit/` | `python-tests`, `python-coverage` |
| Integration | API contracts and module-boundary tests with mocked external systems | `tests/integration/` | `python-tests`, `python-coverage` |
| Component | In-process collaboration across multiple services | `tests/component/` | `python-component-tests` |
| E2E smoke | Template rendering, config catalog, and top-level smoke checks | `tests/e2e/` | `python-e2e-smoke` |

This is the right base framework, so the recommended change is not to introduce a new top-level testing style. The better move is to make the existing pyramid more systematic around hooks.

Build-matrix consistency is covered by `make validate-build` and `tests/unit/tools/test_validate_build_matrix.py`. These checks stay fast and local: they validate `docker-bake.hcl`, Dockerfile coverage, and template `image:` references without requiring a Docker daemon.

## Recommended Test Layout

Mature infrastructure-style Python repositories usually keep the test pyramid, but make unit tests more domain-oriented so contributors know where a new test belongs without reading the whole suite. Dayu can move in that direction incrementally without breaking CI:

| Layer | Recommended grouping | Why it fits Dayu |
| --- | --- | --- |
| Unit | `tests/unit/core_lib/` for runtime library contracts, `tests/unit/runtime_services/` for monitor/processor/controller service contracts, `tests/unit/backend/` for control-plane orchestration contracts, `tests/unit/` root for legacy or cross-cutting unit tests | Dayu has both reusable runtime helpers and long-lived service shells; separating shared library, backend control plane, and runtime services keeps unit tests easier to navigate |
| Integration | `tests/integration/` grouped by API or cross-module boundary | Backend/server contracts already map well to this |
| Component | `tests/component/` grouped by pipeline slice or multi-service collaboration | Best place to prove generator/controller/processor interplay |
| E2E smoke | `tests/e2e/` for template/catalog/render smoke checks only | Keeps end-to-end tests fast and low-maintenance |

The important part is not a big-bang file move. New tests can start using `tests/unit/core_lib/` and `tests/unit/runtime_services/` immediately, and existing tests can migrate gradually. This is fully compatible with the current `pytest` discovery and therefore with both GitHub Actions and CircleCI, because CI only calls the existing `make test-unit-integration`, `make test-component`, `make test-e2e`, and `make coverage-python` targets.

The `core_lib` unit tree should now mirror the production package layout more directly:

| Production area | Test home |
| --- | --- |
| `dependency/core/lib/common/` | `tests/unit/core_lib/common/` |
| `dependency/core/lib/content/` | `tests/unit/core_lib/content/` |
| `dependency/core/lib/network/` | `tests/unit/core_lib/network/` |
| `dependency/core/lib/scheduling/` | `tests/unit/core_lib/scheduling/` |
| `dependency/core/lib/estimation/` | `tests/unit/core_lib/estimation/` |
| `dependency/core/lib/solver/` | `tests/unit/core_lib/solver/` |
| `dependency/core/lib/algorithms/` | `tests/unit/core_lib/algorithms/` |

This mirrors the structure used in mature infrastructure-style repositories: tests live near the conceptual ownership boundary of the code they protect, so contributors can add or find a unit test without scanning unrelated packages first.

Within each mirrored area, keep file names and test names behavior-oriented instead of generic. The current preferred pattern is:

- file names like `test_<family>_algorithms.py`, `test_<module>_edge_cases.py`, or `test_<service>_runtime.py`
- test function names like `test_<behavior>_<expected_result>`
- one file per closely related family of hooks or helpers, instead of mixing getters, queues, monitors, and visualizers in a single large file unless they truly share the same contract surface

This is the same tradeoff many mature Python projects make: keep the directory tree aligned with the production ownership boundary, then keep individual test files narrow enough that a contributor can understand intent from the filename alone.

## Hook-Centric Test Matrix

Dayu's most important extension seam is:

`ClassFactory -> Context.get_algorithm() -> algorithm auto-loader -> runtime consumer`

A mature hook test strategy should cover that seam in four layers:

| Scope | What to prove | Suggested home |
| --- | --- | --- |
| Registry contract | Aliases register correctly, duplicates fail, package registration skips private symbols | `tests/unit/` |
| Resolution contract | Env vars and YAML-selected aliases instantiate the expected hook with merged parameters | `tests/unit/` |
| Runtime lifecycle | Generator, backend, monitor, and processor call hooks in the right order and propagate outputs correctly | `tests/unit/` plus `tests/integration/` |
| Cross-component behavior | A chosen hook family changes real runtime behavior without breaking the pipeline | `tests/component/` |

## What Is Covered Now

The repository already had good coverage in these areas:

- Backend control-plane APIs and visualization config validation.
- Datasource behavior for manifest-driven `http_video`.
- Scheduler, processor, controller, and cross-component happy paths.
- Template-level smoke tests and CI segmentation.

The weaker areas were mostly around hook internals:

- `ClassFactory` registration edge cases were not tested directly.
- `Context` path and algorithm resolution branches had limited direct coverage.
- `dependency/core/lib/algorithms/__init__.py` optional-dependency fallback behavior was only implicitly covered.
- `Generator` and `VideoGenerator` hook lifecycle behavior relied mostly on broader component tests.

## Core Lib Audit

For `dependency/core/lib/` outside `algorithms/`, the current state is now much closer to a maintainable baseline:

| Area | Status | Notes |
| --- | --- | --- |
| `common/class_factory.py`, `common/context.py` | Strong direct coverage | Hook registration, lookup, env/config resolution, and error branches are tested directly |
| `common/cache.py`, `common/config.py`, `common/queue.py`, `common/resource.py`, `common/service.py`, `common/utils.py`, `common/record.py`, `common/instance.py` | Strong direct coverage | Runtime helper contracts are covered with isolated unit tests |
| `common/counter.py`, `common/encode_ops.py`, `common/hash_ops.py`, `common/name.py`, `common/video_ops.py`, `common/yaml_ops.py`, `common/file_ops.py` | Direct coverage added | Serialization, naming, media conversion, filesystem helpers, and temp-file lifecycle now have dedicated unit tests |
| `runtime/model.py`, `runtime/context.py`, `runtime/resolver.py`, `runtime/lease.py`, `runtime/task_barrier.py` | Strong direct coverage | Bootstrap parsing, immutable endpoint identities, exact task routes, scheduler-backed lease identities, queryable task barriers, ambiguity rejection, and fail-closed behavior are unit-tested without a Kubernetes client |
| `scheduling/deployment_plan.py` | Strong direct coverage | Canonical service-to-node-list normalization, complete DAG coverage, candidate scoping, explicit cloud identity, and invalid-plan rejection are tested independently from policy implementations |
| `scheduling/source_selection.py` | Strong direct coverage | Strict scope parsing, independent Backend-authorized source candidates, selected/all-edge semantics, and rejection of legacy discovery inputs are tested directly and through Backend install transactions |
| `content/service.py`, `content/dag.py`, `content/task.py` | Strong direct coverage | DAG extraction, service timing, and task lifecycle are exercised directly |
| `network/client.py`, `network/api.py`, `network/utils.py` | Good direct coverage | HTTP behavior includes management-call status/detail preservation while the existing lenient caller contract remains stable; API constants and pure address utilities are covered, and topology and ports are no longer discovered in runtime workers |
| `solver/*` | Good direct coverage | Longest path, LCA, and intermediate node logic have dedicated tests |
| `estimation/time_estimation.py`, `estimation/accuracy_estimation.py`, `estimation/overhead_estimation.py`, `estimation/model_flops_estimation.py` | Direct coverage added | Timing tickets, accuracy math, overhead logs, and FLOPs fallback behavior are now unit-tested |
| `common/log.py`, `common/constant.py`, `network/api.py`, `network/utils.py`, package `__init__.py` exports | Mostly trivial / indirectly covered | These are thin constants or re-export layers and do not need the same density of tests |

`tests/unit/core_lib/test_runtime_kubernetes_boundary.py` is an architectural guard: it parses every runtime Python module and requirement file and fails if a Kubernetes client import, `KubeConfig`, `NodeInfo`, `PortConfig`, `PortInfo`, `force_refresh`, or a legacy processor-Pod deletion hook is reintroduced.

## Runtime Service Contracts

For service-layer code, the most useful unit tests are not “does FastAPI work” or “does OpenCV decode real video.” Mature projects usually focus on consumer contracts instead:

- `processor` unit tests should prove how a task is read, how upstream content is consumed, how model/tracker/classifier dependencies are invoked, and how results/scenarios are written back into the task.
- task-lease tests should prove one Generator acquire per task, no Controller/Processor lease calls, Distributor final
  renew/persist/scenario-ack/release ordering, transient acquire failure isolation, retired-schedule refresh, atomic
  directory commit plus old-lease clamping, inactive-revision renewal rejection without a marker, immutable retirement
  deadlines/forced revocation, task-bound reservation promotion, restart-persistent active records, and fail-closed
  behavior without contacting Kubernetes.
- structured application unit tests should prove each application service can be instantiated independently, returns only
  service-specific `outputs`, and does not encode DAG membership or shared DAG schemas.
- `monitor` unit tests should prove how monitor workers are instantiated, scheduled, joined, and posted to the scheduler API.
- `distributor` unit tests should prove persistence ordering, incremental reads, export behavior, and scheduler forwarding without needing a full pipeline run.
- `generator_server` unit tests should prove context parameters are collected and passed into the selected generator hook correctly.
- `scheduler` unit tests should prove direct single-process Uvicorn startup, thread-pool-safe synchronous Redis handlers,
  startup-policy fallback, backup offloading, scenario/resource propagation, task reservation/admission snapshots,
  exact known-barrier reads, resource-lock passthrough, unchanged structured 4xx responses, and bounded rejection
  logging without depending on policy-specific agents.
- package `__init__` tests should prove optional imports degrade gracefully when third-party dependencies are absent, while real core import errors still surface immediately.
- `*_server` unit tests should prove atomic ordered queue/running snapshots, background handling, serialization, timing
  hooks, retry requeue order, and outbound request contracts.

This is now reflected in `tests/unit/runtime_services/`, which gives Dayu a clearer place for service-shell behavior without pushing everything into slower integration tests.

For backend code, mature open source projects also usually keep orchestration tests close to the control plane instead of mixing them into generic helper tests. Dayu can follow that pattern with `tests/unit/backend/`, where the important contracts are:

- pure RuntimeService rendering and immutable runtime/session models
- fixed-GVR create/watch/delete behavior, including watch expiration and exact status identity binding
- ConfigMap `resourceVersion` compare-and-swap and corruption/conflict handling
- scheduler-first install, activation/publication readback, proposal/atomic-retirement-CAS/return, bounded retirement,
  cleanup fairness while retirement remains continuously pending, UID-guarded `Background` deletion acceptance within
  one shared deadline, asynchronous/repeated/concurrent uninstall admission, and
  generator-first/immediate-fence/Scheduler-admission-fence/worker-delete uninstall ordering
- Scheduler management failures retaining endpoint, status, and structured detail in the existing RuntimeSession and
  `/install_state` contracts instead of collapsing into a generic plan error
- strict deployment-plan validation plus optional cloud-backup composition across initial install and redeploy
- backend-owned node/agent preflight and batched exact-Pod-UID telemetry joins, including all-container Kubernetes
  Quantity aggregation, allocatable/capacity denominator labeling, and fail-closed partial metrics
- polling loops such as result fetching and runtime reconcile control
- single-flight Scheduler/Kubernetes telemetry sampling, independent cadences, immediate route placeholders,
  per-resource available/stale/unavailable states, exact-Pod batch binding, rebind/uninstall race rejection, singleton
  bandwidth projection/conflict handling, and frontend settle-then-schedule/abort contracts
- connection-boundary DNS canonicalization for HTTP, Redis, iperf, simulated datasource, and shell-rendered support
  endpoints while persisted RuntimeDirectory identities remain unchanged
- root-task artifact isolation, atomic publication, and one lifespan-owned lease-TTL cleaner with no branch/Pod cleanup
- exact task-delivery ACK validation, persistent Generator/Processor ownership, and replay-safe multipart retries
- idempotent predecessor barriers retained until the merged next hop acknowledges ownership
- config validation, snapshot export, and state persistence helpers
- backend-only failure handling that should not require component or end-to-end tests

Frontend lifecycle transitions use Node's built-in test runner in `frontend/tests/`. `make frontend-test` covers the
lifecycle projection, request/JSON-body timeouts, HTTP-safe UUID generation, identity-bound install acceptance,
single-flight snapshot application, action-waiter invalidation, stale refresh serialization, cancellation availability,
permanent gateway-error rejection, non-ready metric fencing, failed-session cleanup, and target-bound uninstall
completion across an immediate replacement Session. They also verify that progressing or delayed cleanup keeps the
same uninstall spinner and install lock until the target identity disappears, including in a browser that did not issue
the command. The Python frontend contract tests
additionally verify that components consume the single Pinia lifecycle observer rather than issuing their own
`/install_state` polls.

The Node frontend suite also covers the installation-form preference boundary: drafts are isolated by the namespace
reported by `/install_state`, contain only stable semantic identifiers, survive catalog reordering, reconcile against
current datasource/DAG/node catalogs, and fail open when browser storage is corrupt or unavailable. Static frontend
contracts ensure installation completion does not erase that draft and **Clear** remains the only explicit removal
path.

`tests/unit/test_dayu_shell.py` executes `ACTION=stop` against deterministic fake `kubectl`, HTTP, timeout, and clock
commands. It covers cancellation before any RuntimeService exists, strict lifecycle-JSON validation, target-bound
replacement completion, same-target `cancelling-install`/`preparing-uninstall` waits, trusted targetless-stop
completion, Backend-failure fallback cleanup, idempotent absent
namespaces with cluster-RBAC cleanup, and non-zero results when final namespace removal is present or unverifiable.
These are behavior tests; source-string assertions alone are not sufficient for the stop contract.

So the answer to “is everything outside `algorithms/` covered?” is now: almost all meaningful runtime logic is covered directly. Kubernetes/live-environment failure paths belong exclusively to the backend control plane; runtime worker tests use injected bootstrap and directory snapshots instead of cluster mocks.

The performance boundary is reviewable even without a live cluster: runtime modules cannot import the Kubernetes
package, rendered Pods cannot receive a service-account token, task-routed components cannot resolve from bootstrap,
and backend telemetry tests assert periodic whole-directory batches plus a zero-I/O management read path rather than
per-worker or per-browser calls. A live environment is still
useful for measuring Sedna/EdgeMesh activation latency, but it is not needed to prove that edge Python processes have no
Kubernetes call path.

## Recommended Test Logic

When adding or changing hooks, prefer this order:

1. Unit-test the hook contract itself.
2. Unit-test the consumer that calls the hook.
3. Add one integration or component test only when the hook changes a service boundary or pipeline behavior.
4. Keep E2E tests as smoke checks instead of trying to make them exhaustive.

That balance keeps feedback fast while still protecting the dynamic runtime wiring that makes Dayu flexible.

## Remaining Gaps

Even after strengthening hook tests, these areas are still good future targets:

- Scheduler research agents and policy families under `dependency/core/lib/algorithms/schedule_agent/`.
- More monitor and visualization hook permutations.
- Processor scenario-extraction chains and queue strategies.
- Live-cluster backend control-plane integration checks, if the project later adopts a stable test cluster fixture.
- Frontend unit tests around configuration workflows.
- Real external-system tests for backend-owned RuntimeService lifecycle, if the project later adds a heavier integration environment.
