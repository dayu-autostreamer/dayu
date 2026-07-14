# Hook Catalog

This catalog lists the registered hook implementations currently present in the repository. Alias names are the values
that appear in templates, environment variables, and visualization configs.

## Base Signatures

| Hook type                       | Base signature                                                      |
|---------------------------------|---------------------------------------------------------------------|
| `GEN_BSO`                       | `__call__(system)`                                                  |
| `GEN_ASO`                       | `__call__(system, scheduler_response)`                              |
| `GEN_BSTO`                      | `__call__(system, new_task)`                                        |
| `GEN_GETTER`                    | `__call__(system)`                                                  |
| `GEN_GETTER_FILTER`             | `__call__(system)`                                                  |
| `GEN_FILTER`                    | `__call__(system, frame) -> bool`                                   |
| `GEN_PROCESS`                   | `__call__(system, frame, source_resolution, target_resolution)`     |
| `GEN_COMPRESS`                  | `__call__(system, frame_buffer, file_name)`                         |
| `SCH_CONFIG_EXTRACTION`         | `__call__(scheduler)`                                               |
| `SCH_SCENARIO_RETRIEVAL`        | `__call__(task)`                                                    |
| `SCH_POLICY_RETRIEVAL`          | `__call__(task)`                                                    |
| `SCH_STARTUP_POLICY`            | `__call__(info)`                                                    |
| `SCH_SELECTION_POLICY`          | `__call__(info)`                                                    |
| `SCH_INITIAL_DEPLOYMENT_POLICY` | `__call__(info)`                                                    |
| `SCH_REDEPLOYMENT_POLICY`       | `__call__(info)`                                                    |
| `PRO_SCENARIO`                  | `__call__(result, task)`                                            |
| `MON_PRAM`                      | `__call__()` returning a thread that updates `system.resource_info` |
| `RESULT_VISUALIZER`             | `__call__(task, resource=None)`                                     |
| `SYSTEM_VISUALIZER`             | `__call__(resource=..., scheduling_overhead=...)`                   |

## Generator Hooks

### `GENERATOR`

| Alias   | Module                                         | Purpose                                                                                                            |
|---------|------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| `video` | `dependency/core/generator/video_generator.py` | Main generator loop for video sources. Resolves generator-side data hooks and periodically requests new schedules. |

### `GEN_BSO`

| Alias       | Module                                                                            | Purpose                                                                                                         | Notes                                                       |
|-------------|-----------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------|
| `simple`    | `dependency/core/lib/algorithms/before_schedule_operation/simple_operation.py`    | Build the base scheduler request from source id, metadata, source device, edge device list, and DAG deployment. | Default choice for standard scheduling flows.               |
| `casva`     | `dependency/core/lib/algorithms/before_schedule_operation/casva_operation.py`     | Extend the base request with `skip_count` from the getter filter.                                               | Resets the CASVA getter filter after packaging the request. |
| `chameleon` | `dependency/core/lib/algorithms/before_schedule_operation/chameleon_operation.py` | Extend the base request with an encoded frame and hash code for online profiling.                               | Used by Chameleon's HTTP-video profiling workflow.          |

### `GEN_ASO`

| Alias    | Module                                                                        | Purpose                                                                                             | Notes                                                                 |
|----------|-------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------|
| `simple` | `dependency/core/lib/algorithms/after_schedule_operation/simple_operation.py` | Apply a scheduler plan to generator state, or fall back to all-local execution when no plan exists. | Updates `meta_data`, `task_dag`, and cached service deployment state. |
| `casva`  | `dependency/core/lib/algorithms/after_schedule_operation/casva_operation.py`  | Same as `simple`, but ensures a default `qp` exists in metadata.                                    | Used by CASVA's encoder-aware schedule decisions.                     |

### `GEN_BSTO`

| Alias       | Module                                                                               | Purpose                                                                           | Notes                                                 |
|-------------|--------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|-------------------------------------------------------|
| `simple`    | `dependency/core/lib/algorithms/before_submit_task_operation/simple_operation.py`    | No-op hook.                                                                       | Good default when no extra task enrichment is needed. |
| `cevas`     | `dependency/core/lib/algorithms/before_submit_task_operation/cevas_operation.py`     | Record compressed file size into task temporary data.                             | Used by the CEVAS scheduler family.                   |
| `casva`     | `dependency/core/lib/algorithms/before_submit_task_operation/casva_operation.py`     | Record file size and estimate content dynamics relative to the previous config.   | Used by CASVA reward and scenario logic.              |
| `chameleon` | `dependency/core/lib/algorithms/before_submit_task_operation/chameleon_operation.py` | Cache the first encoded frame and first hash code for the next scheduler request. | Enables Chameleon profiling.                          |
| `steady`    | `dependency/core/lib/algorithms/before_submit_task_operation/steady_operation.py`    | Record file size and keep the task shape expected by Steady-family baselines.     | Used by steady-style policy templates.                 |

### `GEN_GETTER`

| Alias        | Module                                                            | Purpose                                                           | Notes                                                                                                                             |
|--------------|-------------------------------------------------------------------|-------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| `http_video` | `dependency/core/lib/algorithms/data_getter/http_video_getter.py` | Pull buffered clips from the simulated HTTP datasource service.   | Uses `/source` then `/file` on `datasource/http_video.py`.                                                                        |
| `rtsp_video` | `dependency/core/lib/algorithms/data_getter/rtsp_video_getter.py` | Read frames directly from an RTSP stream and build tasks locally. | Typically consumes streams produced by `datasource/rtsp_video.py`; handles reconnects and offloads compression into a subprocess. |
| `v4l2_video` | `dependency/core/lib/algorithms/data_getter/v4l2_video_getter.py` | Read frames from a local V4L2 camera device and build tasks.      | Used by real-camera datasource configs such as `config/datasource_configs/real_camera.yaml`.                                      |

### `GEN_GETTER_FILTER`

| Alias    | Module                                                                      | Purpose                                                                   | Notes                                            |
|----------|-----------------------------------------------------------------------------|---------------------------------------------------------------------------|--------------------------------------------------|
| `simple` | `dependency/core/lib/algorithms/data_getter_filter/simple_getter_filter.py` | Always allow the generator to fetch the next batch.                       | Default behavior.                                |
| `casva`  | `dependency/core/lib/algorithms/data_getter_filter/casva_getter_filter.py`  | Skip fetch rounds when arrivals are too delayed and count skipped rounds. | Produces `skip_count` for CASVA scheduler input. |
| `scheduler_permitted` | `dependency/core/lib/algorithms/data_getter_filter/scheduler_permitted_getter_filter.py` | Fetch only when the scheduler permits a new generation round. | Used by scheduler-driven and Hedger policy templates. |

### `GEN_FILTER`

| Alias     | Module                                                          | Purpose                                                                          | Notes                                                                   |
|-----------|-----------------------------------------------------------------|----------------------------------------------------------------------------------|-------------------------------------------------------------------------|
| `simple`  | `dependency/core/lib/algorithms/frame_filter/simple_filter.py`  | Downsample frames by comparing raw FPS and target FPS.                           | Uses skip or remain intervals derived from the configured FPS.          |
| `dynamic` | `dependency/core/lib/algorithms/frame_filter/dynamic_filter.py` | Adapt frame acceptance over time using random FPS ranges and smooth transitions. | Experimental; time-varying behavior rather than content-aware behavior. |
| `motion`  | `dependency/core/lib/algorithms/frame_filter/motion_filter.py`  | Adapt target FPS according to measured motion ratio in the scene.                | Experimental; motion-aware filter using background subtraction.         |

### `GEN_PROCESS`

| Alias      | Module                                                             | Purpose                                                                  | Notes                                                     |
|------------|--------------------------------------------------------------------|--------------------------------------------------------------------------|-----------------------------------------------------------|
| `simple`   | `dependency/core/lib/algorithms/frame_process/simple_process.py`   | Resize frames when source and target resolutions differ.                 | Default and stable path.                                  |
| `adaptive` | `dependency/core/lib/algorithms/frame_process/adaptive_process.py` | Extract foreground regions, compute ROIs, and emit ROI sidecar metadata. | Experimental and tied to region-aware encoding workflows. |

### `GEN_COMPRESS`

| Alias      | Module                                                               | Purpose                                                                        | Notes                                                                           |
|------------|----------------------------------------------------------------------|--------------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| `simple`   | `dependency/core/lib/algorithms/frame_compress/simple_compress.py`   | Write the buffered frames directly to a video file using the configured codec. | Default and stable path.                                                        |
| `casva`    | `dependency/core/lib/algorithms/frame_compress/casva_compress.py`    | Encode frames, then re-encode with FFmpeg and a scheduler-selected `qp`.       | Used by CASVA.                                                                  |
| `adaptive` | `dependency/core/lib/algorithms/frame_compress/adaptive_compress.py` | Perform ROI-aware, RL-guided encoding for adaptive video transmission.         | Experimental; depends on extra model files and a specialized encoder toolchain. |

## Scheduler Hooks

### `SCH_CONFIG_EXTRACTION`

| Alias       | Module                                                                                     | Purpose                                                                                        |
|-------------|--------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|
| `simple`    | `dependency/core/lib/algorithms/schedule_config_extraction/simple_config_extraction.py`    | Load `fps`, `resolution`, `buffer_size`, and pipeline knobs from the default scheduler config. |
| `fc`        | `dependency/core/lib/algorithms/schedule_config_extraction/fc_config_extraction.py`        | Load resolution-only knob space for the feedback controller family.                            |
| `casva`     | `dependency/core/lib/algorithms/schedule_config_extraction/casva_config_extraction.py`     | Load `fps`, `resolution`, `qp`, plus CASVA DRL and hyper-parameter files.                      |
| `chameleon` | `dependency/core/lib/algorithms/schedule_config_extraction/chameleon_config_extraction.py` | Load `fps` and `resolution` knob space for Chameleon profiling.                                |
| `deepva`    | `dependency/core/lib/algorithms/schedule_config_extraction/deepva_config_extraction.py`    | Load DeepVA-specific scheduler assets and defaults.                                             |
| `hei`       | `dependency/core/lib/algorithms/schedule_config_extraction/hei_config_extraction.py`       | Load HEI knob spaces plus DRL and hyper-parameter files.                                       |
| `hei_drl`   | `dependency/core/lib/algorithms/schedule_config_extraction/hei_drl_config_extraction.py`   | Same role as `hei`, but from the `scheduler/hei-drl` asset directory.                          |
| `hedger`    | `dependency/core/lib/algorithms/schedule_config_extraction/hedger_config_extraction.py`    | Load Hedger network, hyper, and agent configs.                                                 |

### `SCH_SCENARIO_RETRIEVAL`

| Alias    | Module                                                                                    | Purpose                                                                                         |
|----------|-------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|
| `simple` | `dependency/core/lib/algorithms/schedule_scenario_retrieval/simple_scenario_retrieval.py` | Build scheduler state from the first scenario record and average delay per buffered item.       |
| `casva`  | `dependency/core/lib/algorithms/schedule_scenario_retrieval/casva_scenario_retrieval.py`  | Extend scenario retrieval with transmit delay, segment size, content dynamics, and buffer size. |
| `steady` | `dependency/core/lib/algorithms/schedule_scenario_retrieval/steady_scenario_retrieval.py` | Build the scenario state expected by Steady-family scheduling baselines.                        |

### `SCH_POLICY_RETRIEVAL`

| Alias    | Module                                                                                 | Purpose                                                                    |
|----------|----------------------------------------------------------------------------------------|----------------------------------------------------------------------------|
| `simple` | `dependency/core/lib/algorithms/schedule_policy_retrieval/simple_policy_extraction.py` | Reconstruct the currently applied metadata and DAG deployment from a task. |

### `SCH_STARTUP_POLICY`

| Alias   | Module                                                                           | Purpose                                                                                     |
|---------|----------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| `fixed` | `dependency/core/lib/algorithms/schedule_startup_policy/fixed_startup_policy.py` | Return a default all-cloud policy with `720p`, `5 fps`, `buffer_size=4`, and inherited DAG. |

### `SCH_SELECTION_POLICY`

| Alias    | Module                                                                                | Purpose                                               | Notes                                    |
|----------|---------------------------------------------------------------------------------------|-------------------------------------------------------|------------------------------------------|
| `fixed`  | `dependency/core/lib/algorithms/schedule_selection_policy/fixed_selection_policy.py`  | Choose a source node by exact position or hostname.   | Supports `selected_edge_nodes` / `all_edge_nodes`; invalid or unavailable fixed values fail instead of falling back. |
| `random` | `dependency/core/lib/algorithms/schedule_selection_policy/random_selection_policy.py` | Choose a source node randomly from the Backend-authorized source set. | Supports `selected_edge_nodes` / `all_edge_nodes`; it performs no topology discovery. |

### `SCH_INITIAL_DEPLOYMENT_POLICY`

| Alias    | Module                                                                                                  | Purpose                                                            | Notes                                                                 |
|----------|---------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------|-----------------------------------------------------------------------|
| `fixed`  | `dependency/core/lib/algorithms/schedule_initial_deployment_policy/fixed_initial_deployment_policy.py`  | Apply a fixed deployment map from inline config or a mounted file. | Emits only current-DAG services and rejects missing, empty, or non-candidate placements. |
| `cloud`  | `dependency/core/lib/algorithms/schedule_initial_deployment_policy/cloud_initial_deployment_policy.py` | Explicitly deploy every current-DAG service to `system.cloud_device`. | Used by `cloud-only-policy`; does not depend on an empty-list fallback. |
| `full`   | `dependency/core/lib/algorithms/schedule_initial_deployment_policy/full_initial_deployment_policy.py`   | Deploy all services to all selected nodes.                         | Simple high-availability baseline.                                    |
| `random` | `dependency/core/lib/algorithms/schedule_initial_deployment_policy/random_initial_deployment_policy.py` | Randomly distribute services across selected nodes.                | Supports optional `max_service_num`.                                  |
| `hedger` | `dependency/core/lib/algorithms/schedule_initial_deployment_policy/hedger_initial_deployment_policy.py` | Ask the Hedger subsystem for an initial deployment plan.           | Falls back to default deployment when Hedger does not produce a plan. |
| `hedger-deployment-only` | `dependency/core/lib/algorithms/schedule_initial_deployment_policy/hedger_deployment_only_initial_deployment_policy.py` | Use the Hedger deployment-only ablation for initial placement. | Shares Hedger deployment plumbing while constraining the ablation scope. |
| `hedger-flat` | `dependency/core/lib/algorithms/schedule_initial_deployment_policy/hedger_flat_initial_deployment_policy.py` | Use the flat Hedger ablation for initial placement. | Research/benchmark-oriented variant. |
| `hedger-no-graph-encoder` | `dependency/core/lib/algorithms/schedule_initial_deployment_policy/hedger_no_graph_encoder_initial_deployment_policy.py` | Use the no-graph-encoder Hedger ablation for initial placement. | Neutralizes learned graph embeddings while preserving the Hedger interface. |
| `hedger-offloading-only` | `dependency/core/lib/algorithms/schedule_initial_deployment_policy/hedger_offloading_only_initial_deployment_policy.py` | Use the Hedger offloading-only ablation for initial placement. | Research/benchmark-oriented variant. |

### `SCH_REDEPLOYMENT_POLICY`

| Alias    | Module                                                                                      | Purpose                                                              | Notes                                                       |
|----------|---------------------------------------------------------------------------------------------|----------------------------------------------------------------------|-------------------------------------------------------------|
| `fixed`  | `dependency/core/lib/algorithms/schedule_redeployment_policy/fixed_redeployment_policy.py`  | Apply a fixed redeployment map from inline config or a mounted file. | Emits only current-DAG services and rejects missing, empty, or non-candidate placements. |
| `cloud`  | `dependency/core/lib/algorithms/schedule_redeployment_policy/cloud_redeployment_policy.py` | Keep every current-DAG service explicitly on `system.cloud_device`. | Used by `cloud-only-policy`; the returned hostname is exact. |
| `non`    | `dependency/core/lib/algorithms/schedule_redeployment_policy/non_redeployment_policy.py`    | Keep the active processor deployment from `RuntimeDirectory`.        | No-op redeployment strategy; performs no cluster discovery. |
| `hedger` | `dependency/core/lib/algorithms/schedule_redeployment_policy/hedger_redeployment_policy.py` | Ask the Hedger subsystem for a redeployment plan.                    | Falls back to default deployment when no plan is available. |
| `deepva` | `dependency/core/lib/algorithms/schedule_redeployment_policy/deepva_redeployment_policy.py` | Apply DeepVA redeployment behavior. | Used by the DeepVA policy family. |
| `dynamic` | `dependency/core/lib/algorithms/schedule_redeployment_policy/dynamic_redeployment_policy.py` | Convert the latest exact offloading decision into a current-DAG deployment plan. | Uses one snapshot plus an explicit validated fallback; it does not poll or assume a cloud hostname. |
| `offline_profiling` | `dependency/core/lib/algorithms/schedule_redeployment_policy/offline_profiling_redeployment_policy.py` | Use offline latency/profile information for redeployment. | Covers every current-DAG service and uses the injected cloud identity when no edge candidate exists. |
| `online_profiling` | `dependency/core/lib/algorithms/schedule_redeployment_policy/online_profiling_redeployment_policy.py` | Use online profiling feedback for redeployment. | Paired with `online_profiling` agent templates. |
| `latency_matrix_collector` | `dependency/core/lib/algorithms/schedule_redeployment_policy/latency_matrix_collector_redeployment_policy.py` | Collect or apply latency-matrix-oriented redeployment decisions. | Template exists even though it is not currently in `scheduler_policies.yaml`. |
| `hedger-deployment-only` | `dependency/core/lib/algorithms/schedule_redeployment_policy/hedger_deployment_only_redeployment_policy.py` | Hedger deployment-only redeployment variant. | Research/benchmark-oriented variant. |
| `hedger-flat` | `dependency/core/lib/algorithms/schedule_redeployment_policy/hedger_flat_redeployment_policy.py` | Flat Hedger redeployment variant. | Research/benchmark-oriented variant. |
| `hedger-no-graph-encoder` | `dependency/core/lib/algorithms/schedule_redeployment_policy/hedger_no_graph_encoder_redeployment_policy.py` | No-graph-encoder Hedger redeployment variant. | Research/benchmark-oriented variant. |
| `hedger-offloading-only` | `dependency/core/lib/algorithms/schedule_redeployment_policy/hedger_offloading_only_redeployment_policy.py` | Hedger offloading-only redeployment variant. | Research/benchmark-oriented variant. |

### `SCH_AGENT`

| Alias       | Module                                                                   | Purpose                                                                                        | Notes                                                                                        |
|-------------|--------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| `cloud`     | `dependency/core/lib/algorithms/schedule_agent/cloud_agent.py`           | Keep the source stage on its selected edge and schedule every non-source DAG stage on `system.cloud_device`. | Used by `cloud-only-policy`; no cloud hostname is hard-coded.                                 |
| `fixed`     | `dependency/core/lib/algorithms/schedule_agent/fixed_agent.py`           | Apply fixed configuration and fixed offloading decisions.                                      | Static baseline policy.                                                                      |
| `fc`        | `dependency/core/lib/algorithms/schedule_agent/fc_agent.py`              | Feedback controller that adjusts resolution based on a sliding delay window.                   | Implements the Feedback Controlling policy family.                                           |
| `steady`    | `dependency/core/lib/algorithms/schedule_agent/steady_agent.py`          | Steady baseline that searches configuration and pipeline split from resource/scenario context. | Execution profiles use `execute_role=edge|cloud`; no Kubernetes node hostname is assumed.    |
| `madeye`    | `dependency/core/lib/algorithms/schedule_agent/madeye_agent.py`          | MadEye-style policy search over FPS, resolution, buffer size, and edge service count.          | Uses feedback-aware search assets.                                                           |
| `adamec`    | `dependency/core/lib/algorithms/schedule_agent/adamec_agent.py`          | AdaMEC-style policy search over configuration and pipeline split.                              | Uses mounted knowledge-base assets.                                                          |
| `gecko`     | `dependency/core/lib/algorithms/schedule_agent/gecko_agent.py`           | Gecko-style policy search over configuration and pipeline split.                               | Uses mounted knowledge-base assets.                                                          |
| `cevas`     | `dependency/core/lib/algorithms/schedule_agent/cevas_agent.py`           | Predict and choose a single pipeline split point between edge and cloud.                       | Driven by object-count and file-size history.                                                |
| `casva`     | `dependency/core/lib/algorithms/schedule_agent/casva_agent.py`           | DRL-based configuration agent over resolution, FPS, QP, and segment size.                      | Uses scenario, content-dynamics, and reward computation tied to transmit delay and accuracy. |
| `chameleon` | `dependency/core/lib/algorithms/schedule_agent/chameleon_agent.py`       | Online profiler that ranks candidate configs from recent raw frames and estimated F1 scores.   | Only supported for `http_video` sources.                                                     |
| `deepva`    | `dependency/core/lib/algorithms/schedule_agent/deepva_agent.py`          | DeepVA scheduler family agent.                                                                 | Uses DeepVA assets and replay/state logic.                                                   |
| `dynamic`   | `dependency/core/lib/algorithms/schedule_agent/dynamic_agent.py`         | Select execution devices from bandwidth and edge-device load.                                  | Experimental heuristic baseline.                                                             |
| `offline_profiling` | `dependency/core/lib/algorithms/schedule_agent/offline_profiling_agent.py` | Choose offloading targets from bandwidth, offline latency profiles, and queue pressure. | Experimental profiling baseline. |
| `online_profiling` | `dependency/core/lib/algorithms/schedule_agent/online_profiling_agent.py` | Choose offloading targets using online profiling feedback. | Experimental profiling baseline. |
| `latency_matrix_collector` | `dependency/core/lib/algorithms/schedule_agent/latency_matrix_collector_agent.py` | Collect latency-matrix-oriented scheduling observations. | Template exists outside the current installable policy catalog. |
| `hei`       | `dependency/core/lib/algorithms/schedule_agent/hei_agent.py`             | Hierarchical embodied intelligence agent with macro DRL and micro negative feedback control.   | Maintains a per-source state buffer from scenario and resource updates.                      |
| `hei_nf`    | `dependency/core/lib/algorithms/schedule_agent/hei_nf_agent.py`          | Micro-only negative-feedback version of HEI.                                                   | Uses latest policy plus latest task delay.                                                   |
| `hei_drl`   | `dependency/core/lib/algorithms/schedule_agent/hei_drl_agent.py`         | Macro-only DRL version of HEI.                                                                 | Chooses resolution, FPS, buffer size, and pipeline partition.                                |
| `hei_syn`   | `dependency/core/lib/algorithms/schedule_agent/hei_synchronous_agent.py` | Synchronous HEI variant that couples macro DRL decisions with micro negative feedback updates. | Keeps separate overhead estimators for macro and micro stages.                               |
| `hedger`    | `dependency/core/lib/algorithms/schedule_agent/hedger_agent.py`          | Hedger-based scheduler agent for topology-aware deployment and scheduling.                     | Advanced subsystem backed by `dependency/core/lib/algorithms/shared/hedger/`.                |
| `hedger-deployment-only` | `dependency/core/lib/algorithms/schedule_agent/hedger_deployment_only_agent.py` | Hedger ablation focused on deployment behavior. | Research/benchmark-oriented variant. |
| `hedger-flat` | `dependency/core/lib/algorithms/schedule_agent/hedger_flat_agent.py` | Flat Hedger ablation with a collapsed agent structure. | Research/benchmark-oriented variant. |
| `hedger-no-graph-encoder` | `dependency/core/lib/algorithms/schedule_agent/hedger_no_graph_encoder_agent.py` | Hedger ablation that disables learned graph encoder restoration. | Research/benchmark-oriented variant. |
| `hedger-offloading-only` | `dependency/core/lib/algorithms/schedule_agent/hedger_offloading_only_agent.py` | Hedger ablation focused on offloading behavior. | Research/benchmark-oriented variant. |

## Processor Hooks

### `PROCESSOR`

| Alias                        | Module                                                    | Purpose                                                             | Notes                                                    |
|------------------------------|-----------------------------------------------------------|---------------------------------------------------------------------|----------------------------------------------------------|
| `detector_processor`         | `dependency/core/processor/detector_processor.py`         | Run detection on all frames in a task file.                         | Writes unified `bbox` output records and scenario data.  |
| `detector_tracker_processor` | `dependency/core/processor/detector_tracker_processor.py` | Detect on the first frame, then track on subsequent frames.         | Writes unified `bbox` output records and scenario data.  |
| `classifier_processor`       | `dependency/core/processor/classifier_processor.py`       | Classify ROIs produced by a previous stage.                         | Reads `bbox` records and writes unified `text` records.  |
| `roi_classifier_processor`   | `dependency/core/processor/roi_classifier_processor.py`   | ROI-aware classification with per-ROI ids and cache reset per task. | Reads `bbox` records and writes unified `text` records.  |
| `structured_processor`       | `dependency/core/processor/structured_processor.py`       | Run an independent structured application service.                  | Passes unified upstream content to the service.          |

### `PRO_QUEUE`

| Alias    | Module                                                      | Purpose                                  | Notes                                                                                                   |
|----------|-------------------------------------------------------------|------------------------------------------|---------------------------------------------------------------------------------------------------------|
| `simple` | `dependency/core/lib/algorithms/task_queue/simple_queue.py` | FIFO queue with no admission control.    | Default behavior.                                                                                       |
| `limit`  | `dependency/core/lib/algorithms/task_queue/limit_queue.py`  | FIFO queue with bounded growth behavior. | When over the configured size, it drops roughly half of the queued items before appending the new task. |

### `PRO_SCENARIO`

| Alias          | Module                                                                             | Purpose                                                    | Notes                                    |
|----------------|------------------------------------------------------------------------------------|------------------------------------------------------------|------------------------------------------|
| `obj_num`      | `dependency/core/lib/algorithms/scenario_extraction/object_number_extraction.py`   | Count objects per frame.                                   | Used by several scheduler families.      |
| `obj_size`     | `dependency/core/lib/algorithms/scenario_extraction/object_size_extraction.py`     | Compute mean object area ratio per frame.                  | Depends on task metadata resolution.     |
| `obj_velocity` | `dependency/core/lib/algorithms/scenario_extraction/object_velocity_extraction.py` | Placeholder alias reserved for object-velocity extraction. | Current implementation is not completed. |
| `structured_profile` | `dependency/core/lib/algorithms/scenario_extraction/structured_profile_extraction.py` | Store the processor-created `profile` dictionary as scenario data. | Used by structured processor services; profile currently contains only `frame_count`. |

## Monitor Hooks

### `MON_PRAM`

| Alias                 | Module                                                                            | Purpose                                                                            |
|-----------------------|-----------------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| `cpu_usage`           | `dependency/core/lib/algorithms/parameter_monitor/cpu_usage_monitor.py`           | Report host CPU utilization via `psutil`.                                          |
| `memory_usage`        | `dependency/core/lib/algorithms/parameter_monitor/memory_usage_monitor.py`        | Report host memory utilization via `psutil`.                                       |
| `memory_capacity`     | `dependency/core/lib/algorithms/parameter_monitor/memory_capacity_monitor.py`     | Report total host memory capacity in GB.                                           |
| `available_bandwidth` | `dependency/core/lib/algorithms/parameter_monitor/available_bandwidth_monitor.py` | Measure cloud-edge bandwidth using `iperf3` and a scheduler-managed resource lock. |
| `queue_length`        | `dependency/core/lib/algorithms/parameter_monitor/queue_length_monitor.py`        | Query queue lengths through exact local processor routes from `RuntimeDirectory`.  |
| `model_flops`         | `dependency/core/lib/algorithms/parameter_monitor/model_flops_monitor.py`         | Query model FLOPs through exact local processor routes from `RuntimeDirectory`.    |
| `model_memory`        | `dependency/core/lib/algorithms/parameter_monitor/model_memory_monitor.py`        | Query processor-reported RSS through exact routes and retain the observed maximum. |
| `cpu_flops`           | `dependency/core/lib/algorithms/parameter_monitor/cpu_flops_monitor.py`           | Estimate host CPU peak FLOPs from `lscpu`.                                         |
| `gpu_flops`           | `dependency/core/lib/algorithms/parameter_monitor/gpu_flops_monitor.py`           | Estimate GPU FLOPs using CUDA device metadata.                                     |
| `gpu_usage`           | `dependency/core/lib/algorithms/parameter_monitor/gpu_usage_monitor.py`           | Report GPU usage using NVML, `nvidia-smi`, Jetson sysfs, or `tegrastats`.          |

## Visualization Hooks

### `RESULT_VISUALIZER`

| Alias                      | Module                                                                                    | Purpose                                                                       |
|----------------------------|-------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|
| `frame`                    | `dependency/core/lib/algorithms/result_visualizer/frame_visualizer.py`                    | Extract and return the first video frame as a base64 image.                   |
| `bbox_frame`               | `dependency/core/lib/algorithms/result_visualizer/bbox_frame_visualizer.py`               | Draw structured `bbox` output records onto the frame.                         |
| `roi_frame`                | `dependency/core/lib/algorithms/result_visualizer/roi_frame_visualizer.py`                | Draw ROI bounding boxes onto the first frame.                                 |
| `roi_label_frame`          | `dependency/core/lib/algorithms/result_visualizer/roi_label_frame_visualizer.py`          | Draw ROI bounding boxes plus labels from a downstream service.                |
| `multiple_roi_frame`       | `dependency/core/lib/algorithms/result_visualizer/multiple_roi_frame_visualizer.py`       | Draw bounding boxes from multiple ROI-producing services onto the same frame. |
| `segmentation_frame`       | `dependency/core/lib/algorithms/result_visualizer/segmentation_frame_visualizer.py`       | Draw structured segmentation polygons or polylines.                           |
| `track_frame`              | `dependency/core/lib/algorithms/result_visualizer/track_frame_visualizer.py`              | Draw structured track histories and boxes.                                    |
| `trajectory_frame`         | `dependency/core/lib/algorithms/result_visualizer/trajectory_frame_visualizer.py`         | Draw structured predicted trajectory points.                                  |
| `pose_frame`               | `dependency/core/lib/algorithms/result_visualizer/pose_frame_visualizer.py`               | Draw structured keypoint and pose outputs.                                    |
| `text_frame`               | `dependency/core/lib/algorithms/result_visualizer/text_frame_visualizer.py`               | Draw structured text outputs, optionally anchored to another service.         |
| `event_frame`              | `dependency/core/lib/algorithms/result_visualizer/event_frame_visualizer.py`              | Draw graph/event summaries from structured risk outputs.                      |
| `obj_num`                  | `dependency/core/lib/algorithms/result_visualizer/object_number_visualizer.py`            | Render mean object count as a curve value.                                    |
| `multiple_obj_num`         | `dependency/core/lib/algorithms/result_visualizer/multiple_object_number_visualizer.py`   | Render mean object counts for requested DAG services as curve values.         |
| `e2e_delay`                | `dependency/core/lib/algorithms/result_visualizer/end_to_end_delay_visualizer.py`         | Render total task delay as a curve value.                                     |
| `service_processing_delay` | `dependency/core/lib/algorithms/result_visualizer/service_processing_delay_visualizer.py` | Render per-service execution time for requested DAG nodes.                    |
| `service_queue_length`     | `dependency/core/lib/algorithms/result_visualizer/service_queue_length_visualizer.py`     | Join Task exact processor routes with the backend-prefetched resource snapshot. |
| `service_gantt`            | `dependency/core/lib/algorithms/result_visualizer/service_gantt_visualizer.py`            | Render task execute intervals on service lanes; `services` defaults to all business DAG services. |
| `service_device_gantt`     | `dependency/core/lib/algorithms/result_visualizer/service_device_gantt_visualizer.py`     | Render one service's task execute intervals on its deployed-device lanes.     |
| `dag_deployment`           | `dependency/core/lib/algorithms/result_visualizer/dag_deployment_topology_visualizer.py`  | Render the deployment topology of the current DAG.                            |
| `dag_offloading`           | `dependency/core/lib/algorithms/result_visualizer/dag_offloading_topology_visualizer.py`  | Render the current offloading targets of DAG services.                        |

### `SYSTEM_VISUALIZER`

| Alias               | Module                                                                             | Purpose                                            |
|---------------------|------------------------------------------------------------------------------------|----------------------------------------------------|
| `cpu_usage`         | `dependency/core/lib/algorithms/system_visualizer/cpu_usage_visualizer.py`         | Render CPU usage from one backend-prefetched resource snapshot.    |
| `memory_usage`      | `dependency/core/lib/algorithms/system_visualizer/memory_usage_visualizer.py`      | Render memory usage from one backend-prefetched resource snapshot. |
| `schedule_overhead` | `dependency/core/lib/algorithms/system_visualizer/schedule_overhead_visualizer.py` | Render backend-prefetched scheduler overhead in milliseconds.      |

## Configuration Cheat Sheet

| Hook family                         | Typical config key                                                                                                                    |
|-------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------|
| Generator data path                 | `GEN_FILTER_NAME`, `GEN_PROCESS_NAME`, `GEN_COMPRESS_NAME`, `GEN_GETTER_NAME`, `GEN_GETTER_FILTER_NAME`                               |
| Generator schedule lifecycle        | `GEN_BSO_NAME`, `GEN_ASO_NAME`, `GEN_BSTO_NAME`                                                                                       |
| Scheduler core                      | `SCH_CONFIG_EXTRACTION_NAME`, `SCH_SCENARIO_RETRIEVAL_NAME`, `SCH_POLICY_RETRIEVAL_NAME`, `SCH_STARTUP_POLICY_NAME`, `SCH_AGENT_NAME` |
| Scheduler source/deployment helpers | `SCH_SELECTION_POLICY_NAME`, `SCH_INITIAL_DEPLOYMENT_POLICY_NAME`, `SCH_REDEPLOYMENT_POLICY_NAME`                                     |
| Processor                           | `PROCESSOR_NAME`, `STRUCTURED_PROCESSOR_PARAMETERS`, `PRO_QUEUE_NAME`, `SCENARIOS_EXTRACTORS`                                         |
| Monitor                             | `MONITORS`                                                                                                                            |
| Visualization                       | `hook_name` and optional `hook_params` in visualization YAML                                                                          |

## Maintenance Rules

- When adding a new alias, update this file in the same change.
- When removing or renaming an alias, update templates and config examples at the same time.
- When a hook is experimental, prototype-only, or incomplete, keep that note in the catalog so operators do not mistake
  it for a fully productionized default.
