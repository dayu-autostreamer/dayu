# Documentation

This directory contains the repository-managed technical documentation for Dayu. The goal is to keep the docs close to
the implementation, make internal service contracts discoverable, and document the hook system that drives Dayu's
dynamic behavior.

## Structure

| Section                                       | Description                                                                                          |
|-----------------------------------------------|------------------------------------------------------------------------------------------------------|
| [`api/`](./api/README.md)                     | Component-facing API references for the backend control plane and the internal runtime services.     |
| [`architecture/`](./architecture/README.md)   | System mental model, control-plane/runtime flow, and extension seams.                                |
| [`configuration/`](./configuration/README.md) | How templates, catalogs, env vars, datasource configs, and visualization configs shape a deployment. |
| [`datasource/`](./datasource/README.md)       | Datasource dataset layout, manifest schema, and runtime behavior for `http_video` and `rtsp_video`.  |
| [`development/`](./development/README.md)     | Repository map, contributor workflows, and where to implement common kinds of changes.               |
| [`hooks/`](./hooks/README.md)                 | Hook system overview, configuration model, lifecycle, and extension guidance.                        |
| [`hooks/catalog.md`](./hooks/catalog.md)      | Alias-by-alias catalog of registered hook implementations and their roles.                           |
| [`testing/`](./testing/README.md)             | Test pyramid, hook-focused coverage strategy, and recommended placement for new tests.               |

## Scope

These docs describe the implementation currently present in this repository. They are based on the code under
`backend/`, `dependency/core/`, `datasource/`, and `template/`.

The API documents cover two different contract types:

| Contract type              | Audience                                                                                      | Stability                                                                                 |
|----------------------------|-----------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| Backend control-plane APIs | Frontend, operators, deployment tooling                                                       | Higher-level and operator-facing                                                          |
| Runtime service APIs       | Dayu internal components such as generator, controller, scheduler, processor, and distributor | Internal contracts; keep backward compatibility only when required by deployed components |

The hook documents cover the dynamic extension mechanism used across generator, scheduler, processor, monitor, and
visualization pipelines.

## Reading Order

1. Start with [`architecture/README.md`](./architecture/README.md) for the system-level mental model.
2. Read [`configuration/README.md`](./configuration/README.md) to understand how policies, templates, and env vars
   become a running deployment.
3. Read [`api/README.md`](./api/README.md) for the service map and route references.
4. Read [`datasource/README.md`](./datasource/README.md) if you are changing datasource manifests, `http_video`,
   `rtsp_video`, or clip indexing behavior.
5. Read [`hooks/README.md`](./hooks/README.md) and then [`hooks/catalog.md`](./hooks/catalog.md) if you are changing
   scheduling policies, generators, monitors, processors, or visualization plugins.
6. Read [`development/README.md`](./development/README.md) if you are implementing a feature or refactor and need to
   know where the code lives.
7. Read [`testing/README.md`](./testing/README.md) if you are expanding coverage, adjusting CI test scopes, or adding
   new hooks and want to place tests consistently.
