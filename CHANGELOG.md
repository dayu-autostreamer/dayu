# Dayu Release Notes

---

## v1.3 (In Development)

### Features

### Bug Fix
- Fix iptables rule accumulation for edgemesh in incorrect dayu shutdowns with `dayu.sh` script correction.

### Minor Update

---

## v1.2

### Features
- Add a cyclic redeployment mechanism of service processors for further flexible task processing.
- Add new services to construct a logical topology, including pedestrian-detection, vehicle-detection, 
exposure-identification, category-classification and license-plate-recognition.

### Bug Fix
- Fix concurrency conflicts for starting up multiple streams in rtsp video and http video (`datasource`).
- Fix frame index updating error in http video source (`datasource`).
- Fix port occupation bug of rtsp video datasource after shutting down (`datasource`).
- Support dynamic variable update in visualization modules (`frontend`).
- Support cloud-only deployment of processors (`backend`).
- Fix task forwarding bug in inconsistent deployment and offloading decisions (`controller`).
- Fix possible database locking in concurrency access of distributor (`distributor`).
- Add cache TTL of Kubernetes configurations to avoid additional expense in `PortInfo` and `KubeConfig`.

### Minor Update
- Update more flexible visualization modules to switch different user-defined configurations in multi-stream scenarios (`frontend` / `backend`).
- Clean up frontend code and beatify frontend pages (`frontend`).
- Add persistent storage of installation configuration (`frontend`).
- Add system visualization to monitor system parameters, including resource usage, scheduling cost and so on (`frontend` / `backend` / `scheduler`).
- Improving a fine-grained monitoring architecture including monitoring cpu and gpu flops (`monitor`).
- Add 'USE_TENSORRT' option in processors to choose whether using tensorrt mode (`processor`).
- Add model flops calculation in processors to support model flops monitoring (`processor`).
- Update 'ImagePullPolicy' in template files from 'IfNotPresent' to 'Always' to ensure pulling the latest images.
- Update detailed overhead time statistics in scheduling logs (`scheduler`).
- Accelerate backend visualization through caching mechanism (`backend`).
- Add ROI ID to detection/tracking applications to support roi-accelerated classification applications (`processor`).
- Add ROI Classifier to accelerate roi-level classification applications (`processor`).
- Optimize local service processor by sharing temporary directory on local devices (`controller`/`processor`).
- Add adjustable request scheduling interval in generator to avoid frequent scheduling (`generator`).
- Add temporary file cleaning mechanism in local controller to avoid disk occupation (`controller`).
- Add compatible docker image building for jp4/jp5/jp6 for processor in Nvidia Jetson devices (`processor`/`monitor`).
- Update scenario data structure and processing (`processor`).

---

## v1.1

### Breaking Changes
The basic structure of tasks in dayu is updated from linear pipeline to topological dag (directed acyclic graph) to support more complicated application scenarios.

### Features
- A brand-new forwarding mechanism in the dayu system for tasks with dag structure, including splitting nodes with forking and merging nodes with redis.
- A fine-grained and flexible deployment and offloading mechanism for topological logic nodes and physical nodes, which separates the process of model deployment and task offloading and allows collaboration among multi-edges and cloud.
- A more flexible visualization module in frontend to display customized visualization views for system analysis.
- Add our work on model evolution, adaptively switch models based on scenarios. [(link)](template/scheduler/model-switch.yaml)
- Add our work on video encoding: CRAVE (Collaborative Region-aware Adaptive Video Encoding). It is a region-adaptive video encoding algorithm for cloud-edge collaborative object detection. [(link)](template/scheduler/crave.yaml)

### Bug Fix
- Fix problem of write queue full in rtsp datasource server (`datasource`).
- Fix possible task loss in the system (`controller` / `distributor`).
- Add optional cloud/edge parameters filling in template files for flexible parameter specification in cloud-edge pods.

### Minor Update
- Add cloud and edge template supplementary to support heterogeneous parameters (`backend`).
- Beatify frontend pages (`frontend`).
- Refactor template directory to simplify file structure.
- Unify the base image for system components. 
- Add service of age classification. (Current available services: car-detection, face-detection, gender-classification, age-classification)

---

## v1.0

### Features
- Complete online processing, scheduling and displaying flow of video analytics pipelines.
- Compatible with different operations among the whole flow with various hook functions.
- Easy to deploy on distributed systems and scalable to heterogeneous devices based on KubeEdge.
- Support heterogeneous hook function extensions for research of different topics (like data configuration, task offloading, video encoding, and so on) and implementation of different methods (for baseline comparison).
- Include our latest work on video configuration and task offloading: hierarchical-EI, a two-phase hierarchical scheduling framework based on Embodied Intelligence. It helps adjust system configuration with low cost and high scenario adaption.  [(link)](template/scheduler/hei.yaml)

