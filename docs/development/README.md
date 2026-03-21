# Development Guide

This guide is for contributors who need to change code in the repository and want to quickly find the right implementation area, test layer, and related docs.

## Repository Map

| Path | Purpose | Typical changes |
| --- | --- | --- |
| `backend/` | Backend control plane, deployment orchestration, visualization and log export | API changes, install/query lifecycle, visualization config handling |
| `frontend/` | Vue-based UI for DAG management, deployment, runtime visualization | operator workflows, forms, dashboards, routing |
| `datasource/` | Datasource supervisor, `http_video`, `rtsp_video`, dataset loader | source playback behavior, manifests, source-process lifecycle |
| `components/` | Container-facing entrypoints for runtime services | packaging or service bootstrap changes |
| `dependency/core/controller/` | Task ingress and return-path orchestration | controller behavior, transport timing |
| `dependency/core/distributor/` | Result persistence, incremental queries, export | database behavior, result lifecycle |
| `dependency/core/generator/` | Source-side task generation and schedule requests | task segmentation, source loop behavior |
| `dependency/core/monitor/` | Resource sampling loop | monitor orchestration |
| `dependency/core/processor/` | Processor service shells and inference orchestration | processor behavior, queueing, scenario extraction |
| `dependency/core/scheduler/` | Scheduler shell and per-source agent orchestration | runtime scheduling behavior |
| `dependency/core/lib/` | Shared runtime library: hooks, content model, network helpers, estimators | reusable runtime helpers and most extensibility points |
| `dependency/core/applications/` | Concrete AI application implementations | detector, classifier, tracker, service-specific logic |
| `template/` | Deployment composition and default runtime env | scheduler families, processor templates, default visualizers |
| `config/` | Example datasource and visualization inputs | sample runtime inputs and demos |
| `docs/` | Repository-managed technical documentation | architecture, API, hooks, datasource, testing, contributor docs |
| `tests/` | Unit, integration, component, and e2e tests | regression coverage |
| `tools/` | Small developer and operations utilities | offline tooling, reporting helpers |

## Main Runtime Entry Points

Most component containers are intentionally thin. The logic lives under `dependency/core/` and the container entrypoints under `components/` usually just expose an ASGI app or call a runtime loop.

| Entrypoint | Delegates to |
| --- | --- |
| `components/scheduler/main.py` | `dependency/core/scheduler/SchedulerServer` |
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
- `backend/backend_core.py` for orchestration and state management
- `backend/template_helper.py` if deployment rendering changes
- `docs/api/backend.md` if the route contract changes
- `tests/unit/` or `tests/integration/` for coverage

### Add or change a hook

Usually touch:

- `dependency/core/lib/algorithms/<family>/`
- `dependency/core/lib/common/class_factory.py` only if a new hook family is needed
- one or more templates under `template/` to expose the new alias
- [`../hooks/README.md`](../hooks/README.md) and [`../hooks/catalog.md`](../hooks/catalog.md)
- unit tests under `tests/unit/`

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
make lint-python
make python-syntax
make test-unit-integration
make test-component
make test-e2e
make coverage-python
make frontend-install
make frontend-check
```

For test-layer guidance, see [`../testing/README.md`](../testing/README.md).

## Documentation Maintenance Rules

Repository quality improves fastest when docs stay close to code. For Dayu, treat docs updates as part of the feature:

| Change type | Update these docs |
| --- | --- |
| Route or response changes | `docs/api/` |
| Hook lifecycle, aliases, or parameters | `docs/hooks/` |
| Datasource manifest or playback changes | `docs/datasource/` |
| Repository workflow, test strategy, or contributor path changes | `docs/development/` or `docs/testing/` |
| Big-picture architecture or deployment composition changes | `docs/architecture/` and `docs/configuration/` |

## Suggested Reading By Task

| If you are doing this... | Read this first |
| --- | --- |
| understanding the platform at a high level | [`../architecture/README.md`](../architecture/README.md) |
| changing templates or env-driven behavior | [`../configuration/README.md`](../configuration/README.md) |
| modifying backend or runtime APIs | [`../api/README.md`](../api/README.md) |
| adding a policy or hook | [`../hooks/README.md`](../hooks/README.md) |
| adding tests | [`../testing/README.md`](../testing/README.md) |
