# Documentation

This directory contains the repository-managed technical documentation for Dayu. The goal is to keep the docs close to the implementation, make internal service contracts discoverable, and document the hook system that drives Dayu's dynamic behavior.

## Structure

| Section | Description |
| --- | --- |
| [`api/`](./api/README.md) | Component-facing API references for the backend control plane and the internal runtime services. |
| [`datasource/`](./datasource/README.md) | Datasource dataset layout, manifest schema, and runtime behavior for `http_video` and `rtsp_video`. |
| [`hooks/`](./hooks/README.md) | Hook system overview, configuration model, lifecycle, and extension guidance. |
| [`hooks/catalog.md`](./hooks/catalog.md) | Alias-by-alias catalog of registered hook implementations and their roles. |

## Scope

These docs describe the implementation currently present in this repository. They are based on the code under `backend/`, `dependency/core/`, `datasource/`, and `template/`.

The API documents cover two different contract types:

| Contract type | Audience | Stability |
| --- | --- | --- |
| Backend control-plane APIs | Frontend, operators, deployment tooling | Higher-level and operator-facing |
| Runtime service APIs | Dayu internal components such as generator, controller, scheduler, processor, and distributor | Internal contracts; keep backward compatibility only when required by deployed components |

The hook documents cover the dynamic extension mechanism used across generator, scheduler, processor, monitor, and visualization pipelines.

## Reading Order

1. Start with [`api/README.md`](./api/README.md) for the system-level service map.
2. Read [`api/backend.md`](./api/backend.md) if you are working on frontend or operator workflows.
3. Read [`api/runtime-services.md`](./api/runtime-services.md) if you are modifying generator, scheduler, controller, processor, distributor, or datasource behavior.
4. Read [`datasource/README.md`](./datasource/README.md) if you are changing datasource manifests, `http_video`, `rtsp_video`, or clip indexing behavior.
5. Read [`hooks/README.md`](./hooks/README.md) and then [`hooks/catalog.md`](./hooks/catalog.md) if you are changing scheduling policies, generators, monitors, processors, or visualization plugins.
