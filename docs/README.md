# Dayu Repository Documentation

This directory contains the repository-managed technical documentation for Dayu. It is written for contributors and
operators who need to understand the implementation that is currently in this repository.

## Website and Repository Docs

Dayu has two documentation surfaces with different jobs:

| Surface | Primary job | Good content |
| --- | --- | --- |
| [Project documentation site](https://dayu-autostreamer.github.io/docs/) | Public onboarding and tutorial flow | Why Dayu, architecture narrative, installation preparation, first run, UI walkthroughs, case studies, community pages. |
| This repository `docs/` directory | Implementation reference close to code | API contracts, template/config details, hook aliases, datasource contracts, lifecycle behavior, testing and maintainer guidance. |

When a topic is mostly about teaching a first-time user how to use Dayu, keep the full walkthrough on the website and
link back to repository references only for exact contracts. When a topic changes with code, templates, or tests, keep
the authoritative details here.

## Start Here

| Reader goal | Start here | Why |
| --- | --- | --- |
| New to Dayu as a user | [Documentation site](https://dayu-autostreamer.github.io/docs/) | Public introduction, preparation, tutorial, developer guide, and case-study flow. |
| New to the repository | [`repository-quickstart.md`](./repository-quickstart.md) | Code-facing reading path, local checks, and implementation landmarks. |
| Need the vocabulary | [`concepts.md`](./concepts.md) | Defines DAGs, services, processor templates, task content, policies, hooks, datasources, and queries. |
| Operating a running Dayu system | [`operations/`](./operations/README.md) | Managed-runtime prerequisites, RBAC, install/publication, rollout drain, safe stop, and useful checks. |
| Changing code | [`development/`](./development/README.md) | Repository map, common change workflows, and docs/tests to update with code changes. |
| Adding coverage | [`testing/`](./testing/README.md) | Test pyramid and where new tests should live. |

## Reference Map

| Section | Description |
| --- | --- |
| [`architecture/`](./architecture/README.md) | Control-plane ownership, RuntimeService activation, RuntimeDirectory/task-route flow, and extension seams. |
| [`configuration/`](./configuration/README.md) | How templates, catalogs, env vars, datasource configs, visualization configs, and deployment knobs shape a runtime install. |
| [`configuration/structured-traffic-services.md`](./configuration/structured-traffic-services.md) | Structured traffic service contract and the reviewable driving-risk DAG. |
| [`api/`](./api/README.md) | Backend control-plane APIs plus RuntimeDirectory, lease, and internal runtime service APIs. |
| [`datasource/`](./datasource/README.md) | Dataset layout, manifest schema, and frame-index behavior for source playback. |
| [`hooks/`](./hooks/README.md) | Hook system overview, configuration model, lifecycle, and extension guidance. |
| [`hooks/catalog.md`](./hooks/catalog.md) | Alias-by-alias catalog of registered hook implementations and their roles. |

## Scope

These docs describe the implementation currently present in this repository. They are based on the code under
`backend/`, `dependency/core/`, `datasource/`, and `template/`. They should not become a second copy of the public
tutorial site.

The API documents cover two different contract types:

| Contract type              | Audience                                                                                      | Stability                                                                                 |
|----------------------------|-----------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| Backend control-plane APIs | Frontend, operators, deployment tooling                                                       | Higher-level and operator-facing                                                          |
| Runtime service APIs       | Dayu internal components such as generator, controller, scheduler, processor, and distributor | Internal contracts; keep backward compatibility only when required by deployed components |

The hook documents cover the dynamic extension mechanism used across generator, scheduler, processor, monitor, and
visualization pipelines.

## Reading Order

1. If you are a first-time Dayu user, start from the [documentation site](https://dayu-autostreamer.github.io/docs/).
2. If you are changing this repository, start with [`repository-quickstart.md`](./repository-quickstart.md).
3. Read [`concepts.md`](./concepts.md) to align vocabulary with code and templates.
4. Read [`architecture/README.md`](./architecture/README.md) for the system-level mental model.
5. Read [`configuration/README.md`](./configuration/README.md) to understand how policies, templates, and env vars
   become a running deployment.
6. Read [`api/README.md`](./api/README.md) for the service map and route references.
7. Read [`hooks/README.md`](./hooks/README.md) and then [`hooks/catalog.md`](./hooks/catalog.md) if you are changing
   scheduling policies, generators, monitors, processors, or visualization plugins.
8. Read [`operations/README.md`](./operations/README.md) when debugging install, query, redeployment, or cleanup behavior.
9. Read [`testing/README.md`](./testing/README.md) before adding or moving coverage.
