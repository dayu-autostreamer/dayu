English | [简体中文](./README_zh.md)

# Dayu

[![Version](https://img.shields.io/github/v/release/dayu-autostreamer/dayu?cacheSeconds=3600)](https://github.com/dayu-autostreamer/dayu/releases)
[![CI](https://github.com/dayu-autostreamer/dayu/actions/workflows/ci.yml/badge.svg)](https://github.com/dayu-autostreamer/dayu/actions/workflows/ci.yml)
[![CircleCI Project](https://circleci.com/gh/dayu-autostreamer/dayu.svg?style=svg)](https://app.circleci.com/pipelines/gh/dayu-autostreamer/dayu)
[![Codecov](https://codecov.io/gh/dayu-autostreamer/dayu/graph/badge.svg)](https://codecov.io/gh/dayu-autostreamer/dayu)
[![Licence](https://img.shields.io/github/license/dayu-autostreamer/dayu.svg)](LICENSE)
[![Homepage](https://img.shields.io/website?url=https%3A%2F%2Fdayu-autostreamer.github.io%2F&label=homepage)](https://dayu-autostreamer.github.io/)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/dayu-autostreamer/dayu/badge)](https://scorecard.dev/viewer/?uri=github.com/dayu-autostreamer/dayu)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/10523/badge)](https://www.bestpractices.dev/projects/10523)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)

![](static/dayu-logo-horizontal.svg)

| <img src="static/news.svg" width="8%">  NEWS                                                                        |
|---------------------------------------------------------------------------------------------------------------------|
| (**Latest Release**) Jul. 2nd, 2026. Dayu v1.3 Release. Please check the [CHANGELOG](CHANGELOG.md#v13) for details. |
| Jan. 15th, 2026. Dayu v1.2 Release. Please check the [CHANGELOG](CHANGELOG.md#v12) for details.                     |
| May. 7th, 2025. Dayu v1.1 Release. Please check the [CHANGELOG](CHANGELOG.md#v11) for details.                      |
| Feb. 21th, 2025. Dayu v1.0 Release. Please check the [CHANGELOG](CHANGELOG.md#v10) for details.                     |

Dayu is a cloud-edge stream analytics platform for deploying, scheduling, and operating DAG-based AI
pipelines across heterogeneous nodes. It combines a backend control plane, a Vue frontend, simulated datasources, and a
runtime collaboration layer for generator, scheduler, controller, processor, distributor, and monitor services.

## Why Dayu

- DAG-based multi-stage stream analytics with dynamic task configuration and offloading
- Hook-based runtime extension for generator, scheduler, processor, monitor, and visualization behavior
- Built-in deployment composition using policy catalogs, component templates, and processor service templates
- Backend-managed result visualization, system telemetry snapshots, and compressed log export
- Compatibility with heterogeneous cloud and edge nodes through a KubeEdge and Sedna oriented deployment model

## Architecture at a Glance

Dayu is composed of five layers:

- **Basic System Layer**: This layer adopts the `KubeEdge` architecture and is deployed on all distributed nodes across
  the cloud-edge environment. `KubeEdge` is the `Kubernetes` extension proposed by Huawei for edge scenarios and can be
  well deployed on devices with limited resources and low performance.
- **Intermediate Interface Layer**: This layer exposes Sedna `RuntimeService` as the immutable application-worker API
  and uses EdgeMesh's exact Service/Pod identity acknowledgement as the activation barrier.
- **System Support Layer**: This layer is designed to offer interactive ui (frontend), automatic installation (backend),
  and simulation datasource (datasource) for users.
- **Collaboration Scheduling Layer**: This layer is composed of functional components independently developed by us to
  complete functions such as pipeline task execution and scheduling collaboration.
- **Application Service Layer**: This layer accepts user-defined service applications. As long as the user develops
  service according to the interface requirements defined by the platform, it can be embedded in the platform as a
  container and complete execution across cloud-edge nodes.

![](static/dayu-layer-structure.png)

## Tutorials and Guides

To get detailed instructions about our dayu system, please refer to the documentation on
the [homepage](https://dayu-autostreamer.github.io/).

Please refer to our [quick start tutorial](https://dayu-autostreamer.github.io/docs/) for a quick start
of the dayu system.

If you want to further develop dayu for your needs, please refer to
our [development tutorial](https://dayu-autostreamer.github.io/docs/developer-guide/how-to-develop).

## Implementation Documentation

Dayu uses two documentation surfaces:

- the [project documentation site](https://dayu-autostreamer.github.io/docs/) for public onboarding, tutorials,
  preparation, UI walkthroughs, case studies, and community pages
- repository docs under [`docs/`](docs/README.md) for implementation-facing API contracts, templates, hooks,
  datasource behavior, lifecycle details, tests, and maintainer guidance

| If you want to...                                             | Start here                                                                                          |
|---------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| follow end-user deployment tutorials                          | [project documentation site](https://dayu-autostreamer.github.io/docs/)                             |
| get oriented in this repository                               | [`docs/repository-quickstart.md`](docs/repository-quickstart.md)                                    |
| understand Dayu's core terms and runtime contracts            | [`docs/concepts.md`](docs/concepts.md)                                                              |
| understand the system design                                  | [`docs/architecture/README.md`](docs/architecture/README.md)                                        |
| understand how policies, templates, and env vars fit together | [`docs/configuration/README.md`](docs/configuration/README.md)                                      |
| inspect backend and runtime APIs                              | [`docs/api/README.md`](docs/api/README.md)                                                          |
| operate install, query, redeploy, or cleanup flows            | [`docs/operations/README.md`](docs/operations/README.md)                                            |
| understand the hook model and built-in aliases                | [`docs/hooks/README.md`](docs/hooks/README.md) and [`docs/hooks/catalog.md`](docs/hooks/catalog.md) |
| work on datasource playback and manifests                     | [`docs/datasource/README.md`](docs/datasource/README.md)                                            |
| navigate the repository as a contributor                      | [`docs/development/README.md`](docs/development/README.md)                                          |
| understand test layers and where to add coverage              | [`docs/testing/README.md`](docs/testing/README.md)                                                  |

The repository docs index lives at [`docs/README.md`](docs/README.md).

## Ecosystem

Dayu is designed around the following ecosystem:

- [Docker Container](https://github.com/docker/docker-ce)
- [Kubernetes](https://github.com/kubernetes/kubernetes)
- [KubeEdge](https://github.com/kubeedge/kubeedge)
- [Sedna](https://github.com/kubeedge/sedna)
- [EdgeMesh](https://github.com/kubeedge/edgemesh)
- [TensorRT](https://github.com/NVIDIA/TensorRT)

`dayu-sedna` and `dayu-edgemesh` are released as a matched pair for each supported Dayu version. Use the following
compatibility matrix when installing or upgrading a deployment:

| Dayu version    | dayu-sedna version                                                                                                                         | dayu-edgemesh version                                                                                                                            |
|-----------------|--------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|
| `v1.4`          | [`v1.1`](https://github.com/dayu-autostreamer/dayu-sedna/tree/v1.1)                                                                        | [`v1.1`](https://github.com/dayu-autostreamer/dayu-edgemesh/tree/v1.1)                                                                           |
| `v1.0` - `v1.3` | [`v1.0`](https://github.com/dayu-autostreamer/dayu-sedna/tree/v1.0) or [`v1.1`](https://github.com/dayu-autostreamer/dayu-sedna/tree/v1.1) | [`v1.0`](https://github.com/dayu-autostreamer/dayu-edgemesh/tree/v1.0) or [`v1.1`](https://github.com/dayu-autostreamer/dayu-edgemesh/tree/v1.1) |

`dayu-sedna` and `dayu-edgemesh` must use the same minor version: pair `v1.0` with `v1.0`, or `v1.1` with `v1.1`.

## Local Testing

To align local testing with CI, use the repository-declared toolchain:

- Python `3.8` via [`.python-version`](.python-version)
- Node.js `20` via [`.nvmrc`](.nvmrc)
- `make` as the primary local task entry point

Python developer dependencies are collected in [`requirements-dev.txt`](requirements-dev.txt), and Python lint
configuration lives in [`pyproject.toml`](pyproject.toml).

Use the provided `Makefile` targets for local feedback loops:

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
make check
```

`make validate-build` checks the Docker Bake image matrix against Dockerfiles and deployment templates.
`make check` is the day-to-day aggregate gate. `make coverage-python` mirrors the Python coverage run used by hosted CI.
`make frontend-check` runs formatting, the lifecycle state-machine tests, and the production build.
`make frontend-lint` remains available as a cleanup target while the frontend template debt continues to be reduced
incrementally.

## Repository Layout

| Path               | Purpose                                                                                          |
|--------------------|--------------------------------------------------------------------------------------------------|
| `backend/`         | backend APIs, orchestration, deployment rendering, visualization, and log export                 |
| `frontend/`        | Vue-based control-plane UI                                                                       |
| `datasource/`      | datasource supervisor, HTTP video source, RTSP source, and manifest-driven dataset loader        |
| `components/`      | thin container-facing service entrypoints                                                        |
| `dependency/core/` | runtime services, shared libraries, hook implementations, and application services               |
| `template/`        | deployment catalogs, component templates, processor templates, and default visualization configs |
| `config/`          | sample datasource and visualization inputs                                                       |
| `docs/`            | repository-managed technical documentation                                                       |
| `tests/`           | unit, integration, component, and e2e tests                                                      |
| `tools/`           | developer and operations utilities                                                               |

## Developer Tooling

The repository includes an offline CLI for summarizing exported logs:

```bash
python tools/log_analysis.py --log path/to/exported-log.json.gz
python tools/log_analysis.py --log path/to/exported-log.json.gz --output-format json
python tools/log_analysis.py --log path/to/exported-log.json.gz --output-format full-json --output-file path/to/full-report.json
python tools/log_analysis.py --log path/to/exported-log.json.gz --slo-seconds 2.0
```

## Contact

If you have questions, feel free to reach out to us in the following ways:

- [Lei Xie (lxie@nju.edu.cn)](mailto:lxie@nju.edu.cn)
- [Wenhui Zhou (whzhou@smail.nju.edu.cn)](mailto:whzhou@smail.nju.edu.cn)

## Citation

If you use Dayu or its scheduling/runtime system in your research, please cite the following paper:

```bibtex
@inproceedings{zhou2026hier-ei,
  title = {Tackling the Imbalance in Video Analytics Pipelines with Hierarchical Embodied Intelligence},
  author = {Zhou, Wenhui and Xie, Lei and Ning, Jingyi and Cao, Shuyu and Wu, Hao and Peng, Qinghua and Fan, Long},
  booktitle = {IEEE INFOCOM 2026 - IEEE Conference on Computer Communications},
  year = {2026},
  pages={1--10},
  publisher = {IEEE},
  doi={10.1109/INFOCOM59046.2026.11571610}
}
```

## License

Dayu is licensed under Apache 2.0. See the [LICENSE](LICENSE) file for details.

## Contributing

If you want to contribute code, docs, or tests:

- read [CONTRIBUTING](CONTRIBUTING.md) for the patch and review workflow
- use the repository docs under [`docs/`](docs/README.md) as the implementation-facing reference
- use the Dayu [documentation site](https://dayu-autostreamer.github.io/docs/) for tutorial-oriented content and
  end-user guidance

Thanks for the following contributors:

<a href="https://github.com/dayu-autostreamer/dayu/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=dayu-autostreamer/dayu"  alt=""/>
</a>

Thanks for the following documentation contributors:

<a href="https://github.com/dayu-autostreamer/dayu-autostreamer.github.io/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=dayu-autostreamer/dayu-autostreamer.github.io"  alt=""/>
</a>
