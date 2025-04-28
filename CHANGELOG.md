# Dayu Release Notes

## v1.1

### Breaking Changes
The basic structure of task in dayu is updated from linear pipeline to topological dag (directed acyclic graph) to support more complicated application scenarios.

### Features
- A more flexible visualization module in frontend to display customized visualization views for system analysis.
- Add our latest work on video encoding: CRAVE (Collaborative Region-aware Adaptive Video Encoding). It is a region-adaptive video encoding algorithm for cloud-edge collaborative object detection. [(link)](template/scheduler/crave.yaml)

### Bug Fix
- Fix write queue full in rtsp datasource server (`datasource`).
- Fix possible task loss in system (`controller` / `distributor`).

### Minor Update
- Add cloud and edge template supplementary to support heterogeneous parameters (`backend`).
- Beatify frontend pages (`frontend`).
- Refactor template directory to simplify file structure.


## v1.0

### Features
- Complete online processing, scheduling and displaying flow of video analytics pipelines.
- Compatible with different operations among the whole flow with numerous hook functions.
- Easy to deploy on distributed systems and scalable to heterogeneous devices based on KubeEdge.
- Support heterogeneous hook function extensions for research of different topics (like data configuration, task offloading, video encoding, and so on) and implementation of different methods (for baseline comparison).
- Include our latest work on video configuration and task offloading: hierarchical-EI, a two-phase hierarchical scheduling framework based on Embodied Intelligence. It helps adjust system configuration with low cost and high scenario adaption.  [(link)](template/scheduler/hei.yaml)

