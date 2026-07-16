# Development Guide

This guide is for contributors who need to change code in the repository and want to quickly find the right implementation area, test layer, and related docs.

## Repository Map

| Path | Purpose | Typical changes |
| --- | --- | --- |
| `backend/` | Backend control plane and the only Python Kubernetes client boundary | RuntimeService/session orchestration, API changes, install/query lifecycle, telemetry, visualization |
| `frontend/` | Vue-based UI for DAG management, deployment, runtime visualization | operator workflows, forms, dashboards, routing |
| `datasource/` | Datasource supervisor, `http_video`, `rtsp_video`, dataset loader | source playback behavior, manifests, source-process lifecycle |
| `components/` | Container-facing entrypoints for runtime services | packaging or service bootstrap changes |
| `dependency/core/controller/` | Task ingress and return-path orchestration | controller behavior, transport timing |
| `dependency/core/distributor/` | Result persistence, incremental queries, export | database behavior, result lifecycle |
| `dependency/core/generator/` | Source-side task generation and schedule requests | task segmentation, source loop behavior |
| `dependency/core/monitor/` | Resource sampling loop | monitor orchestration |
| `dependency/core/processor/` | Processor service shells and inference orchestration | processor behavior, queueing, scenario extraction |
| `dependency/core/scheduler/` | Scheduler shell and per-source agent orchestration | runtime scheduling behavior |
| `dependency/core/lib/` | Shared runtime library: hooks, content/task routes, scheduling contracts, pure runtime context/resolver/lease client, network helpers, estimators | reusable runtime helpers and most extensibility points; never Kubernetes discovery |
| `dependency/core/lib/scheduling/` | Stable contracts shared by Backend, Scheduler and scheduling plugins | independent source/processor permissions, deployment-plan shape, normalization, and validation |
| `dependency/core/applications/` | Concrete AI application implementations | detector, classifier, tracker, service-specific logic |
| `template/` | Deployment composition and default runtime env | scheduler families, processor templates, default visualizers |
| `build/` and `docker-bake.hcl` | Dockerfiles plus the declarative image build matrix | image packaging, platform/tag matrix, JetPack build variants |
| `config/` | Example datasource and visualization inputs | sample runtime inputs and demos |
| `docs/` | Repository-managed technical documentation | architecture, API, hooks, datasource, testing, contributor docs |
| `tests/` | Unit, integration, component, and e2e tests | regression coverage |
| `tools/` | Small developer and operations utilities | offline tooling, reporting helpers |

## Main Runtime Entry Points

Most component containers are intentionally thin. The logic lives under `dependency/core/` and the container entrypoints under `components/` usually just expose an ASGI app or call a runtime loop.

| Entrypoint | Delegates to |
| --- | --- |
| `components/scheduler/main.py` | `dependency/core/scheduler/SchedulerServer` through one direct Uvicorn process |
| `components/processor/main.py` | `dependency/core/processor/ProcessorServer` |
| `components/controller/main.py` | `dependency/core/controller/ControllerServer` |
| `components/distributor/main.py` | `dependency/core/distributor/DistributorServer` |
| `components/generator/main.py` | `dependency/core/generator/GeneratorServer` |
| `components/monitor/main.py` | monitor bootstrap under `dependency/core/monitor/` |

This means most behavioral changes should land in the runtime package, not in `components/`.

## Common Change Workflows

### Change backend control-plane behavior

Usually touch:

- `backend/backend_server.py` for route behavior
- `backend/backend_core.py` for operator/query coordination
- `backend/runtime_orchestrator.py` for the transactional install/rollout/bounded-retirement/uninstall state machine
- `backend/runtime_service_client.py` for the fixed Sedna RuntimeService GVR and condition watch
- `backend/runtime_session_store.py` for ConfigMap CAS persistence
- `backend/cluster_client.py` for backend-owned inventory, preflight, and batched Pod metrics
- `backend/template_helper.py` for pure catalog/source normalization
- `backend/runtime_renderer.py` for pure immutable RuntimeService rendering
- `docs/api/backend.md` if the route contract changes
- `tests/unit/backend/` or `tests/integration/` for coverage

Do not add Kubernetes imports to `dependency/core/`, a service-account token to a RuntimeService Pod, or a caller-owned
cache refresh switch. Runtime topology is represented by `RuntimeDirectory` and exact routes copied into `Task`.

### Change runtime routing or rollout

Usually touch:

- `dependency/core/scheduler/runtime_directory.py` and Scheduler APIs for directory CAS/publication
- `dependency/core/scheduler/task_lease.py` for task ownership, immutable retirement deadlines, and forced fencing
- `dependency/core/lib/runtime/` and `content/task.py` for bootstrap, exact route, and task identity contracts
- generator/distributor consumers for acquire-once/final-renew/persist/scenario-ack/release ordering; controller and
  processor must remain outside the lease protocol
- `backend/runtime_orchestrator.py` for proposal/atomic-retirement commit/recovery plus exact-UID-delete ordering
- [`../api/runtime-services.md`](../api/runtime-services.md), [`../operations/README.md`](../operations/README.md), and
  backend/runtime service unit tests

Preserve these invariants: activate replacements and persist exact old-resource ownership before publication; atomically
commit the route CAS, old-revision marker, and lease clamp in Scheduler; persist Scheduler's authoritative deadline;
return without waiting for old tasks; permit at most one lease-protected retirement; never extend its deadline; use
status-only normal reconciliation and reserve `PATCH` for uninstall's immediate fence; advance retirement and exact-UID
cleanup as independent lanes; and keep generator-first/immediate-fence-and-clear/Scheduler-admission-fence/worker-delete
uninstall ordering.

Keep Scheduler single-process because scheduling-agent state is process-local. Run it directly under Uvicorn rather
than a second worker supervisor, and define blocking Redis-backed RuntimeDirectory/lease handlers synchronously so
FastAPI dispatches them to its thread pool. Do not expose worker-heartbeat or Redis-threading tuning as configuration.

### Add or change a hook

Usually touch:

- `dependency/core/lib/algorithms/<family>/`
- `dependency/core/lib/common/class_factory.py` only if a new hook family is needed
- one or more templates under `template/` to expose the new alias
- [`../hooks/README.md`](../hooks/README.md) and [`../hooks/catalog.md`](../hooks/catalog.md)
- unit tests under `tests/unit/`

Deployment policies and schedule agents are plugins, but their public deployment-plan contract is not an algorithm
implementation. Keep that contract in `dependency/core/lib/scheduling/deployment_plan.py`; Scheduler and every policy
family must import the same validator instead of defining local normalization or fallback rules.

### Add or change a processor service

Usually touch:

- `dependency/core/applications/<service>/`
- `template/processor/<service>.yaml`
- `template/services.yaml`
- sometimes `dependency/core/processor/` if a new processor shell type is required
- API or frontend docs only if the service becomes user-facing in a new way

### Change datasource behavior

Usually touch:

- `datasource/datasource_server.py`
- `datasource/http_video.py`, `datasource/rtsp_video.py`, or `datasource/video_dataset.py`
- `config/datasource_configs/*.yaml` or dataset manifests
- [`../datasource/README.md`](../datasource/README.md)

### Change frontend workflows

Usually touch:

- `frontend/src/`
- `backend/backend_server.py` if the workflow requires new data or route shape
- `docs/api/backend.md` if a backend contract changes

### Add or change a Docker image

Usually touch:

- `build/<image>.Dockerfile` for image packaging details
- `docker-bake.hcl` for the image name, Dockerfile path, target platforms, tags, and JetPack variants
- `template/**/*.yaml` only if runtime deployment should point at a new image name
- [`../../build/README.md`](../../build/README.md) if the build workflow changes
- `tests/unit/tools/test_validate_build_matrix.py` when the build-matrix contract itself changes

Run `make validate-build` after changing image names, Dockerfiles, or templates. The deployment templates remain runtime
configuration; they are not the source of truth for how images are built.

## Quality Gates

The repository already has a good baseline for local verification.

### Toolchain

- Python `3.8` via [`.python-version`](../../.python-version)
- Node.js `20` via [`.nvmrc`](../../.nvmrc)
- Python lint configuration in [`pyproject.toml`](../../pyproject.toml)
- Make-based task entrypoints in [`Makefile`](../../Makefile)

### Common commands

```bash
make install-python-dev
make validate-build
make lint-python
make python-syntax
make test-unit-integration
make test-component
make test-e2e
make coverage-python
make frontend-install
make frontend-test
make frontend-check
```

For test-layer guidance, see [`../testing/README.md`](../testing/README.md).

## Documentation Maintenance Rules

Repository quality improves fastest when docs stay close to code. For Dayu, treat docs updates as part of the feature:

Dayu has a public Docusaurus documentation site and this repository `docs/` tree. Keep full end-user walkthroughs,
installation narratives, UI screenshots, case studies, and community pages on the website. Keep code-coupled contracts,
schemas, aliases, lifecycle details, test guidance, and maintainer notes in this repository. If a change needs both,
write the tutorial flow on the website and link to the exact repository reference page for the low-level contract.

| Change type | Update these docs |
| --- | --- |
| Route or response changes | `docs/api/` |
| Hook lifecycle, aliases, or parameters | `docs/hooks/` |
| Datasource manifest or playback changes | `docs/datasource/` |
| Core vocabulary, task envelope, or DAG/service semantics | `docs/concepts.md` and the affected reference doc |
| Install, uninstall, redeployment, retirement/fencing, or cleanup behavior | `docs/operations/README.md` |
| Repository workflow, test strategy, or contributor path changes | `docs/development/` or `docs/testing/` |
| Big-picture architecture or deployment composition changes | `docs/architecture/` and `docs/configuration/` |

## Suggested Reading By Task

| If you are doing this... | Read this first |
| --- | --- |
| understanding the platform at a high level | [`../architecture/README.md`](../architecture/README.md) |
| entering the repository for the first time | [`../repository-quickstart.md`](../repository-quickstart.md) |
| aligning on Dayu terms and runtime contracts | [`../concepts.md`](../concepts.md) |
| changing templates or env-driven behavior | [`../configuration/README.md`](../configuration/README.md) |
| modifying backend or runtime APIs | [`../api/README.md`](../api/README.md) |
| debugging install/query/cleanup behavior | [`../operations/README.md`](../operations/README.md) |
| adding a policy or hook | [`../hooks/README.md`](../hooks/README.md) |
| adding tests | [`../testing/README.md`](../testing/README.md) |
