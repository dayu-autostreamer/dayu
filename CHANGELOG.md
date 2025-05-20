# Dayu Release Notes

## v1.2

### Features

### Bug Fix
- Fix concurrency conflicts for starting up multiple streams in rtsp video and http video (`datasource`).

### Minor Update
- Update more flexible visualization modules to switch different user-defined configurations in multi-stream scenarios (`frontend`).
- Clean up frontend code and beatify frontend pages (`frontend`).

## v1.1

### Breaking Changes
The basic structure of tasks in dayu is updated from linear pipeline to topological dag (directed acyclic graph) to support more complicated application scenarios.

### Features
- A brand-new forwarding mechanism in the dayu system for tasks with dag structure, including splitting nodes with forking and merging nodes with redis.
- A fine-grained and flexible deployment and offloading mechanism for topological logic nodes and physical nodes, which separates the process of model deployment and task offloading and allows collaboration among multi-edges and cloud.
- A more flexible visualization module in frontend to display customized visualization views for system analysis.
- Add our work on model evolution, adaptively switch models based on scenarios. [(link)](template/scheduler/model-switch.yaml)
- Add our latest work on video encoding: CRAVE (Collaborative Region-aware Adaptive Video Encoding). It is a region-adaptive video encoding algorithm for cloud-edge collaborative object detection. [(link)](template/scheduler/crave.yaml)

### Bug Fix
- Fix problem of write queue full in rtsp datasource server (`datasource`).
- Fix possible task loss in the system (`controller` / `distributor`).
- Add optional cloud/edge parameters filling in template files for flexible parameter specification in cloud-edge pods.

### Minor Update
- Add cloud and edge template supplementary to support heterogeneous parameters (`backend`).
- Beatify frontend pages (`frontend`).
- Refactor template directory to simplify file structure.
- Unify the base image for system components. 
- Add application of age classification. (Current available applications: car-detection, face-detection, gender-classification, age-classification)


## v1.0

### Features
- Complete online processing, scheduling and displaying flow of video analytics pipelines.
- Compatible with different operations among the whole flow with various hook functions.
- Easy to deploy on distributed systems and scalable to heterogeneous devices based on KubeEdge.
- Support heterogeneous hook function extensions for research of different topics (like data configuration, task offloading, video encoding, and so on) and implementation of different methods (for baseline comparison).
- Include our latest work on video configuration and task offloading: hierarchical-EI, a two-phase hierarchical scheduling framework based on Embodied Intelligence. It helps adjust system configuration with low cost and high scenario adaption.  [(link)](template/scheduler/hei.yaml)

