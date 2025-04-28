# Dayu Release Notes

## v1.1

### Breaking Changes
The basic structure of task in dayu is updated from linear pipeline to topological dag (directed acyclic graph) to support more complicated application scenarios.

### Features
- A more flexible visualization module in frontend to display customized visualization views for system analysis.

### Bug Fix
- Fix write queue full in rtsp datasource server (`datasource`).
- Fix possible task loss in system (`controller`/`distributor`).

### Minor Update
- Add cloud and edge template supplementary to support heterogeneous parameters (`backend`).
- beatify frontend pages (`frontend`).

## v1.0

### Features
- Complete online processing, scheduling and displaying flow of video analytics pipelines.
- Compatible with different operations among the whole flow with numerous hook functions.
- Easy to deploy on distributed systems and scalable to heterogeneous devices based on KubeEdge.
- Support different scheduling methods, including our latest work: hierarchical-EI.

