# Testing Strategy

This document explains how Dayu's current code structure maps to the test pyramid and where hook-focused tests should live.

## Repository Structure

| Area | Responsibility | Main paths |
| --- | --- | --- |
| Control plane | Backend APIs, install/query orchestration, visualization config handling | `backend/` |
| Runtime services | Generator, scheduler, controller, processor, distributor, monitor | `dependency/core/` |
| Shared runtime library | Hook registry, context resolution, DAG/task types, network helpers, algorithms | `dependency/core/lib/` |
| Source adapters | `http_video`, `rtsp_video`, dataset readers, datasource API | `datasource/` |
| Service entrypoints | Container-facing `main.py` launchers for components | `components/` |
| Deployment templates | YAML-driven runtime composition and hook selection | `template/` |
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

## Recommended Test Layout

Mature infrastructure-style Python repositories usually keep the test pyramid, but make unit tests more domain-oriented so contributors know where a new test belongs without reading the whole suite. Dayu can move in that direction incrementally without breaking CI:

| Layer | Recommended grouping | Why it fits Dayu |
| --- | --- | --- |
| Unit | `tests/unit/core_lib/` for runtime library contracts, `tests/unit/runtime_services/` for monitor/processor/controller service contracts, `tests/unit/` root for legacy or cross-cutting unit tests | Dayu has both reusable runtime helpers and long-lived service shells; splitting those two concerns keeps unit tests easier to navigate |
| Integration | `tests/integration/` grouped by API or cross-module boundary | Backend/server contracts already map well to this |
| Component | `tests/component/` grouped by pipeline slice or multi-service collaboration | Best place to prove generator/controller/processor interplay |
| E2E smoke | `tests/e2e/` for template/catalog/render smoke checks only | Keeps end-to-end tests fast and low-maintenance |

The important part is not a big-bang file move. New tests can start using `tests/unit/core_lib/` and `tests/unit/runtime_services/` immediately, and existing tests can migrate gradually. This is fully compatible with the current `pytest` discovery and therefore with both GitHub Actions and CircleCI, because CI only calls the existing `make test-unit-integration`, `make test-component`, `make test-e2e`, and `make coverage-python` targets.

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
| `common/cache.py`, `common/config.py`, `common/queue.py`, `common/resource.py`, `common/service.py`, `common/utils.py`, `common/health.py`, `common/record.py`, `common/instance.py` | Strong direct coverage | Runtime helper contracts are covered with isolated unit tests |
| `common/counter.py`, `common/encode_ops.py`, `common/hash_ops.py`, `common/name.py`, `common/video_ops.py`, `common/yaml_ops.py`, `common/file_ops.py` | Direct coverage added | Serialization, naming, media conversion, filesystem helpers, and temp-file lifecycle now have dedicated unit tests |
| `common/kube.py` | Good unit coverage, not fully exhaustive | Pod-to-service topology, cache refresh behavior, running-state checks, and metrics parsing are unit-tested; real cluster behavior still belongs to heavier integration tests |
| `content/service.py`, `content/dag.py`, `content/task.py` | Strong direct coverage | DAG extraction, service timing, and task lifecycle are exercised directly |
| `network/client.py`, `network/node.py`, `network/port.py` | Good direct coverage | HTTP requests, node identity parsing, NodePort cache behavior, and service-port lookup are covered |
| `solver/*` | Good direct coverage | Longest path, LCA, and intermediate node logic have dedicated tests |
| `estimation/time_estimation.py`, `estimation/accuracy_estimation.py`, `estimation/overhead_estimation.py`, `estimation/model_flops_estimation.py` | Direct coverage added | Timing tickets, accuracy math, overhead logs, and FLOPs fallback behavior are now unit-tested |
| `common/log.py`, `common/constant.py`, `network/api.py`, `network/utils.py`, package `__init__.py` exports | Mostly trivial / indirectly covered | These are thin constants or re-export layers and do not need the same density of tests |

## Runtime Service Contracts

For service-layer code, the most useful unit tests are not “does FastAPI work” or “does OpenCV decode real video.” Mature projects usually focus on consumer contracts instead:

- `processor` unit tests should prove how a task is read, how upstream content is consumed, how model/tracker/classifier dependencies are invoked, and how results/scenarios are written back into the task.
- `monitor` unit tests should prove how monitor workers are instantiated, scheduled, joined, and posted to the scheduler API.
- `distributor` unit tests should prove persistence ordering, incremental reads, export behavior, and scheduler forwarding without needing a full pipeline run.
- `generator_server` unit tests should prove context parameters are collected and passed into the selected generator hook correctly.
- `*_server` unit tests should prove queueing, background handling, serialization, timing hooks, and outbound request contracts.

This is now reflected in `tests/unit/runtime_services/`, which gives Dayu a clearer place for service-shell behavior without pushing everything into slower integration tests.

So the answer to “is everything outside `algorithms/` covered?” is now: almost all meaningful runtime logic is covered directly, but not literally every environment-specific branch is exhausted. The remaining light spots are mostly Kubernetes/live-environment failure paths and other code that is better protected by integration coverage than by brittle mocks.

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
- Live-cluster Kubernetes integration checks, if the project later adopts a stable test cluster fixture.
- Frontend unit tests around configuration workflows.
- Real external-system tests for Kubernetes or container lifecycle, if the project later adds a heavier integration environment.
