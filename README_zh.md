简体中文 | [English](./README.md)

# 大禹 Dayu

[![Version](https://img.shields.io/github/release/dayu-autostreamer/dayu)](https://github.com/dayu-autostreamer/dayu/releases)
[![CI](https://github.com/dayu-autostreamer/dayu/actions/workflows/ci.yml/badge.svg)](https://github.com/dayu-autostreamer/dayu/actions/workflows/ci.yml)
[![CircleCI Project](https://circleci.com/gh/dayu-autostreamer/dayu.svg?style=svg)](https://app.circleci.com/pipelines/gh/dayu-autostreamer/dayu)
[![Codecov](https://codecov.io/gh/dayu-autostreamer/dayu/graph/badge.svg)](https://codecov.io/gh/dayu-autostreamer/dayu)
[![Licence](https://img.shields.io/github/license/dayu-autostreamer/dayu.svg)](LICENSE)
[![Homepage](https://img.shields.io/website?url=https%3A%2F%2Fdayu-autostreamer.github.io%2F&label=homepage)](https://dayu-autostreamer.github.io/)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/dayu-autostreamer/dayu/badge)](https://scorecard.dev/viewer/?uri=github.com/dayu-autostreamer/dayu)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/10523/badge)](https://www.bestpractices.dev/projects/10523)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)

![](static/dayu-logo-horizontal.svg)

| <img src="static/news.svg" width="8%">  NEWS                           |
|------------------------------------------------------------------------|
| (**最新版本**) 2026.1.15. 大禹 v1.2 发布，详细信息请查阅 [CHANGELOG](CHANGELOG.md#v12) |
| 2025.5.7. 大禹 v1.1 发布，详细信息请查阅 [CHANGELOG](CHANGELOG.md#v11)             |
| 2025.2.21. 大禹 v1.0 发布，详细信息请查阅 [CHANGELOG](CHANGELOG.md#v10)            |

大禹是一个面向云边协同流式分析的可扩展平台，用于部署、调度和运行基于 DAG 的 AI 服务流水线。它将后端控制面、Vue 前端、模拟数据源、以及
generator / scheduler / controller / processor / distributor / monitor 这组运行时协同组件组合在一起，并通过 hook
机制支持不同调度策略和运行时行为的切换。

## 为什么选择 Dayu

- 支持面向多数据流的 DAG 式 AI 服务流水线
- 通过 hook 机制统一扩展 generator、scheduler、processor、monitor 与 visualization 行为
- 使用模板、策略目录和服务目录来驱动部署，而不是把策略逻辑硬编码在控制面里
- 内置运行结果可视化、系统指标快照与压缩日志导出能力
- 面向异构云边节点设计，可与 KubeEdge 和 Sedna 的部署模型配合

## 架构总览

大禹由五个层次组成：

- **基础系统层**：该层采用 `KubeEdge` 架构，部署在云边环境的所有分布式节点上。`KubeEdge` 是华为为边缘场景提出的
  `Kubernetes` 扩展，能很好地部署在资源有限、性能低下的设备上。
- **中间接口层**：通过修改和扩展官方接口组件 `Sedna` 和通信组件 `Edgemesh`，提供定制化服务安装和组件通信。
- **系统支持层**：为用户提供交互式界面（前端）、自动安装（后端）和模拟数据源（数据源）。
- **协同调度层**：由我们自主开发的功能组件组成，完成流水线任务执行和调度协作等功能。
- **应用服务层**：接收用户定义的服务应用。只要用户根据平台定义的接口需求开发服务，就可以以容器形式嵌入平台并完成云边节点之间的执行。

![](static/dayu-layer-structure.png)

## 教程与指南

请参照我们的[教程](https://dayu-autostreamer.github.io/docs/)来快速尝试大禹系统。

如果你需要进一步开发大禹系统来适应你的需求，请参照我们的[开发教程](https://dayu-autostreamer.github.io/docs/developer-guide/how-to-develop)。

要获取关于大禹系统的详细说明，请参阅[项目主页](https://dayu-autostreamer.github.io/)上的文档。

## 项目文档

仓库内已经开始维护实现导向的技术文档；而公开文档站点仍然更适合教程和上手说明。

| 如果你想...                  | 从这里开始                                                                                             |
|--------------------------|---------------------------------------------------------------------------------------------------|
| 理解系统整体结构                 | [`docs/architecture/README.md`](docs/architecture/README.md)                                      |
| 理解策略、模板与 env 变量如何组成部署    | [`docs/configuration/README.md`](docs/configuration/README.md)                                    |
| 查看后端和运行时 API             | [`docs/api/README.md`](docs/api/README.md)                                                        |
| 理解 hook 机制和内置 alias      | [`docs/hooks/README.md`](docs/hooks/README.md) 与 [`docs/hooks/catalog.md`](docs/hooks/catalog.md) |
| 修改 datasource 与 manifest | [`docs/datasource/README.md`](docs/datasource/README.md)                                          |
| 作为贡献者快速理解仓库结构            | [`docs/development/README.md`](docs/development/README.md)                                        |
| 理解测试分层与新增测试位置            | [`docs/testing/README.md`](docs/testing/README.md)                                                |
| 按教程部署和使用系统               | [项目文档站点](https://dayu-autostreamer.github.io/docs/)                                               |

仓库内文档索引位于 [`docs/README.md`](docs/README.md)。

## 生态依赖

- [Docker Container](https://github.com/docker/docker-ce)
- [Kubernetes](https://github.com/kubernetes/kubernetes)
- [KubeEdge](https://github.com/kubeedge/kubeedge)
- [Sedna](https://github.com/kubeedge/sedna)
- [EdgeMesh](https://github.com/kubeedge/edgemesh)
- [TensorRT](https://github.com/NVIDIA/TensorRT)

与 Dayu 部署模型强相关的配套工程还包括：

- [dayu-sedna](https://github.com/dayu-autostreamer/dayu-sedna)
- [dayu-edgemesh](https://github.com/dayu-autostreamer/dayu-edgemesh)

## 本地测试

为保证本地测试环境与 CI 一致，建议使用仓库中声明的工具链版本：

- Python `3.8`，见 [`.python-version`](.python-version)
- Node.js `20`，见 [`.nvmrc`](.nvmrc)
- 使用 `make` 作为本地开发任务入口

Python 侧开发依赖汇总在 [`requirements-dev.txt`](requirements-dev.txt)，Python lint 配置统一放在 [
`pyproject.toml`](pyproject.toml) 中。

### 常用命令

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
make check
```

`make check` 适合作为日常开发的聚合校验入口；`make coverage-python` 对齐 Codecov 使用的 Python 覆盖率流程；
`make frontend-lint` 仍保留用于逐步清理前端模板历史遗留的 lint 债务。

## 仓库结构

| 路径                 | 用途                                                       |
|--------------------|----------------------------------------------------------|
| `backend/`         | 后端 API、部署编排、可视化与日志导出                                     |
| `frontend/`        | 基于 Vue 的控制面 UI                                           |
| `datasource/`      | datasource supervisor、HTTP 视频源、RTSP 视频源与 manifest 数据集加载器 |
| `components/`      | 运行时服务的容器入口                                               |
| `dependency/core/` | 运行时组件、共享库、hook 实现与应用服务                                   |
| `template/`        | 部署目录、组件模板、processor 模板与默认可视化配置                           |
| `config/`          | datasource 与 visualization 示例输入                          |
| `docs/`            | 仓库内维护的技术文档                                               |
| `tests/`           | unit、integration、component 与 e2e 测试                      |
| `tools/`           | 开发与运维辅助工具                                                |

## 开发工具

仓库提供了一个离线命令行工具，用于汇总分析导出的日志：

```bash
python tools/log_analysis.py --log path/to/exported-log.json.gz
python tools/log_analysis.py --log path/to/exported-log.json.gz --output-format json
```

## 联系我们

如果有任何问题，请随时通过以下方式联系我们：

- [谢磊 lxie@nju.edu.cn](mailto:lxie@nju.edu.cn)
- [周文晖 whzhou@smail.nju.edu.cn](mailto:whzhou@smail.nju.edu.cn)

## 许可证

大禹系统遵循 Apache 2.0 许可证，请参阅 [LICENSE](LICENSE) 获取详细信息。

## 贡献

如果你想贡献代码、文档或测试：

- 请先阅读 [CONTRIBUTING](CONTRIBUTING.md) 了解补丁和 review 流程
- 以仓库内 [`docs/`](docs/README.md) 中的实现文档为主要技术参考
- 以公开文档站点作为教程和终端用户说明的入口

感谢以下开发者对大禹系统的贡献：

<a href="https://github.com/dayu-autostreamer/dayu/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=dayu-autostreamer/dayu"  alt=""/>
</a>

感谢以下开发者对大禹系统文档的贡献：

<a href="https://github.com/dayu-autostreamer/dayu-autostreamer.github.io/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=dayu-autostreamer/dayu-autostreamer.github.io"  alt=""/>
</a>
