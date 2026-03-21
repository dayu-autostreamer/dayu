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
| `RESULT_VISUALIZER`             | `__call__(task)`                                                    |
| `SYSTEM_VISUALIZER`             | `__call__()` or `__call__(resource=...)`                            |

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

### `GEN_GETTER`

| Alias        | Module                                                            | Purpose                                                           | Notes                                                                                                                             |
|--------------|-------------------------------------------------------------------|-------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| `http_video` | `dependency/core/lib/algorithms/data_getter/http_video_getter.py` | Pull buffered clips from the simulated HTTP datasource service.   | Uses `/source` then `/file` on `datasource/http_video.py`.                                                                        |
| `rtsp_video` | `dependency/core/lib/algorithms/data_getter/rtsp_video_getter.py` | Read frames directly from an RTSP stream and build tasks locally. | Typically consumes streams produced by `datasource/rtsp_video.py`; handles reconnects and offloads compression into a subprocess. |

### `GEN_GETTER_FILTER`

| Alias    | Module                                                                      | Purpose                                                                   | Notes                                            |
|----------|-----------------------------------------------------------------------------|---------------------------------------------------------------------------|--------------------------------------------------|
| `simple` | `dependency/core/lib/algorithms/data_getter_filter/simple_getter_filter.py` | Always allow the generator to fetch the next batch.                       | Default behavior.                                |
| `casva`  | `dependency/core/lib/algorithms/data_getter_filter/casva_getter_filter.py`  | Skip fetch rounds when arrivals are too delayed and count skipped rounds. | Produces `skip_count` for CASVA scheduler input. |

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
| `hei`       | `dependency/core/lib/algorithms/schedule_config_extraction/hei_config_extraction.py`       | Load HEI knob spaces plus DRL and hyper-parameter files.                                       |
| `hei_drl`   | `dependency/core/lib/algorithms/schedule_config_extraction/hei_drl_config_extraction.py`   | Same role as `hei`, but from the `scheduler/hei-drl` asset directory.                          |
| `hedger`    | `dependency/core/lib/algorithms/schedule_config_extraction/hedger_config_extraction.py`    | Load Hedger network, hyper, and agent configs.                                                 |

### `SCH_SCENARIO_RETRIEVAL`

| Alias    | Module                                                                                    | Purpose                                                                                         |
|----------|-------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|
| `simple` | `dependency/core/lib/algorithms/schedule_scenario_retrieval/simple_scenario_retrieval.py` | Build scheduler state from the first scenario record and average delay per buffered item.       |
| `casva`  | `dependency/core/lib/algorithms/schedule_scenario_retrieval/casva_scenario_retrieval.py`  | Extend scenario retrieval with transmit delay, segment size, content dynamics, and buffer size. |

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
| `fixed`  | `dependency/core/lib/algorithms/schedule_selection_policy/fixed_selection_policy.py`  | Choose a source node by fixed position or hostname.   | Supports `fixed_value` and `fixed_type`. |
| `random` | `dependency/core/lib/algorithms/schedule_selection_policy/random_selection_policy.py` | Choose a source node randomly from the candidate set. | Useful for testing or baseline policies. |

### `SCH_INITIAL_DEPLOYMENT_POLICY`

| Alias    | Module                                                                                                  | Purpose                                                            | Notes                                                                 |
|----------|---------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------|-----------------------------------------------------------------------|
| `fixed`  | `dependency/core/lib/algorithms/schedule_initial_deployment_policy/fixed_initial_deployment_policy.py`  | Apply a fixed deployment map from inline config or a mounted file. | Filters deployment targets against the selected node set.             |
| `full`   | `dependency/core/lib/algorithms/schedule_initial_deployment_policy/full_initial_deployment_policy.py`   | Deploy all services to all selected nodes.                         | Simple high-availability baseline.                                    |
| `random` | `dependency/core/lib/algorithms/schedule_initial_deployment_policy/random_initial_deployment_policy.py` | Randomly distribute services across selected nodes.                | Supports optional `max_service_num`.                                  |
| `hedger` | `dependency/core/lib/algorithms/schedule_initial_deployment_policy/hedger_initial_deployment_policy.py` | Ask the Hedger subsystem for an initial deployment plan.           | Falls back to default deployment when Hedger does not produce a plan. |

### `SCH_REDEPLOYMENT_POLICY`

| Alias    | Module                                                                                      | Purpose                                                              | Notes                                                       |
|----------|---------------------------------------------------------------------------------------------|----------------------------------------------------------------------|-------------------------------------------------------------|
| `fixed`  | `dependency/core/lib/algorithms/schedule_redeployment_policy/fixed_redeployment_policy.py`  | Apply a fixed redeployment map from inline config or a mounted file. | Filters targets against the current node set.               |
| `non`    | `dependency/core/lib/algorithms/schedule_redeployment_policy/non_redeployment_policy.py`    | Keep the current deployment returned by `KubeConfig`.                | No-op redeployment strategy.                                |
| `hedger` | `dependency/core/lib/algorithms/schedule_redeployment_policy/hedger_redeployment_policy.py` | Ask the Hedger subsystem for a redeployment plan.                    | Falls back to default deployment when no plan is available. |

### `SCH_AGENT`

| Alias       | Module                                                                   | Purpose                                                                                        | Notes                                                                                        |
|-------------|--------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| `fixed`     | `dependency/core/lib/algorithms/schedule_agent/fixed_agent.py`           | Apply fixed configuration and fixed offloading decisions.                                      | Static baseline policy.                                                                      |
| `fc`        | `dependency/core/lib/algorithms/schedule_agent/fc_agent.py`              | Feedback controller that adjusts resolution based on a sliding delay window.                   | Implements the Feedback Controlling policy family.                                           |
| `cevas`     | `dependency/core/lib/algorithms/schedule_agent/cevas_agent.py`           | Predict and choose a single pipeline split point between edge and cloud.                       | Driven by object-count and file-size history.                                                |
| `casva`     | `dependency/core/lib/algorithms/schedule_agent/casva_agent.py`           | DRL-based configuration agent over resolution, FPS, QP, and segment size.                      | Uses scenario, content-dynamics, and reward computation tied to transmit delay and accuracy. |
| `chameleon` | `dependency/core/lib/algorithms/schedule_agent/chameleon_agent.py`       | Online profiler that ranks candidate configs from recent raw frames and estimated F1 scores.   | Only supported for `http_video` sources.                                                     |
| `hei`       | `dependency/core/lib/algorithms/schedule_agent/hei_agent.py`             | Hierarchical embodied intelligence agent with macro DRL and micro negative feedback control.   | Maintains a per-source state buffer from scenario and resource updates.                      |
| `hei_nf`    | `dependency/core/lib/algorithms/schedule_agent/hei_nf_agent.py`          | Micro-only negative-feedback version of HEI.                                                   | Uses latest policy plus latest task delay.                                                   |
| `hei_drl`   | `dependency/core/lib/algorithms/schedule_agent/hei_drl_agent.py`         | Macro-only DRL version of HEI.                                                                 | Chooses resolution, FPS, buffer size, and pipeline partition.                                |
| `hei_syn`   | `dependency/core/lib/algorithms/schedule_agent/hei_synchronous_agent.py` | Synchronous HEI variant that couples macro DRL decisions with micro negative feedback updates. | Keeps separate overhead estimators for macro and micro stages.                               |
| `hedger`    | `dependency/core/lib/algorithms/schedule_agent/hedger_agent.py`          | Hedger-based scheduler agent for topology-aware deployment and scheduling.                     | Advanced subsystem backed by `dependency/core/lib/algorithms/shared/hedger/`.                |

## Processor Hooks

### `PROCESSOR`

| Alias                        | Module                                                    | Purpose                                                             | Notes                                                    |
|------------------------------|-----------------------------------------------------------|---------------------------------------------------------------------|----------------------------------------------------------|
| `detector_processor`         | `dependency/core/processor/detector_processor.py`         | Run detection on all frames in a task file.                         | Stores scenario data after inference.                    |
| `detector_tracker_processor` | `dependency/core/processor/detector_tracker_processor.py` | Detect on the first frame, then track on subsequent frames.         | Good fit for detector-plus-tracker services.             |
| `classifier_processor`       | `dependency/core/processor/classifier_processor.py`       | Classify ROIs produced by a previous stage.                         | Expects `Classifier` instance from application code.     |
| `roi_classifier_processor`   | `dependency/core/processor/roi_classifier_processor.py`   | ROI-aware classification with per-ROI ids and cache reset per task. | Expects `Roi_Classifier` instance from application code. |

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

## Monitor Hooks

### `MON_PRAM`

| Alias                 | Module                                                                            | Purpose                                                                            |
|-----------------------|-----------------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| `cpu_usage`           | `dependency/core/lib/algorithms/parameter_monitor/cpu_usage_monitor.py`           | Report host CPU utilization via `psutil`.                                          |
| `memory_usage`        | `dependency/core/lib/algorithms/parameter_monitor/memory_usage_monitor.py`        | Report host memory utilization via `psutil`.                                       |
| `memory_capacity`     | `dependency/core/lib/algorithms/parameter_monitor/memory_capacity_monitor.py`     | Report total host memory capacity in GB.                                           |
| `available_bandwidth` | `dependency/core/lib/algorithms/parameter_monitor/available_bandwidth_monitor.py` | Measure cloud-edge bandwidth using `iperf3` and a scheduler-managed resource lock. |
| `queue_length`        | `dependency/core/lib/algorithms/parameter_monitor/queue_length_monitor.py`        | Query per-service processor queue lengths from local processor pods.               |
| `model_flops`         | `dependency/core/lib/algorithms/parameter_monitor/model_flops_monitor.py`         | Query per-service model FLOPs from local processor pods.                           |
| `model_memory`        | `dependency/core/lib/algorithms/parameter_monitor/model_memory_monitor.py`        | Read per-service memory usage from Kubernetes metrics.                             |
| `cpu_flops`           | `dependency/core/lib/algorithms/parameter_monitor/cpu_flops_monitor.py`           | Estimate host CPU peak FLOPs from `lscpu`.                                         |
| `gpu_flops`           | `dependency/core/lib/algorithms/parameter_monitor/gpu_flops_monitor.py`           | Estimate GPU FLOPs using CUDA device metadata.                                     |
| `gpu_usage`           | `dependency/core/lib/algorithms/parameter_monitor/gpu_usage_monitor.py`           | Report GPU usage using NVML, `nvidia-smi`, Jetson sysfs, or `tegrastats`.          |

## Visualization Hooks

### `RESULT_VISUALIZER`

| Alias                      | Module                                                                                    | Purpose                                                                       |
|----------------------------|-------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|
| `frame`                    | `dependency/core/lib/algorithms/result_visualizer/frame_visualizer.py`                    | Extract and return the first video frame as a base64 image.                   |
| `roi_frame`                | `dependency/core/lib/algorithms/result_visualizer/roi_frame_visualizer.py`                | Draw ROI bounding boxes onto the first frame.                                 |
| `roi_label_frame`          | `dependency/core/lib/algorithms/result_visualizer/roi_label_frame_visualizer.py`          | Draw ROI bounding boxes plus labels from a downstream service.                |
| `multiple_roi_frame`       | `dependency/core/lib/algorithms/result_visualizer/multiple_roi_frame_visualizer.py`       | Draw bounding boxes from multiple ROI-producing services onto the same frame. |
| `obj_num`                  | `dependency/core/lib/algorithms/result_visualizer/object_number_visualizer.py`            | Render mean object count as a curve value.                                    |
| `e2e_delay`                | `dependency/core/lib/algorithms/result_visualizer/end_to_end_delay_visualizer.py`         | Render total task delay as a curve value.                                     |
| `service_processing_delay` | `dependency/core/lib/algorithms/result_visualizer/service_processing_delay_visualizer.py` | Render per-service execution time for requested DAG nodes.                    |
| `dag_deployment`           | `dependency/core/lib/algorithms/result_visualizer/dag_deployment_topology_visualizer.py`  | Render the deployment topology of the current DAG.                            |
| `dag_offloading`           | `dependency/core/lib/algorithms/result_visualizer/dag_offloading_topology_visualizer.py`  | Render the current offloading targets of DAG services.                        |

### `SYSTEM_VISUALIZER`

| Alias               | Module                                                                             | Purpose                                            |
|---------------------|------------------------------------------------------------------------------------|----------------------------------------------------|
| `cpu_usage`         | `dependency/core/lib/algorithms/system_visualizer/cpu_usage_visualizer.py`         | Render scheduler-reported CPU usage per device.    |
| `memory_usage`      | `dependency/core/lib/algorithms/system_visualizer/memory_usage_visualizer.py`      | Render scheduler-reported memory usage per device. |
| `schedule_overhead` | `dependency/core/lib/algorithms/system_visualizer/schedule_overhead_visualizer.py` | Render scheduler overhead in milliseconds.         |

## Configuration Cheat Sheet

| Hook family                         | Typical config key                                                                                                                    |
|-------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------|
| Generator data path                 | `GEN_FILTER_NAME`, `GEN_PROCESS_NAME`, `GEN_COMPRESS_NAME`, `GEN_GETTER_NAME`, `GEN_GETTER_FILTER_NAME`                               |
| Generator schedule lifecycle        | `GEN_BSO_NAME`, `GEN_ASO_NAME`, `GEN_BSTO_NAME`                                                                                       |
| Scheduler core                      | `SCH_CONFIG_EXTRACTION_NAME`, `SCH_SCENARIO_RETRIEVAL_NAME`, `SCH_POLICY_RETRIEVAL_NAME`, `SCH_STARTUP_POLICY_NAME`, `SCH_AGENT_NAME` |
| Scheduler source/deployment helpers | `SCH_SELECTION_POLICY_NAME`, `SCH_INITIAL_DEPLOYMENT_POLICY_NAME`, `SCH_REDEPLOYMENT_POLICY_NAME`                                     |
| Processor                           | `PROCESSOR_NAME`, `PRO_QUEUE_NAME`, `SCENARIOS_EXTRACTORS`                                                                            |
| Monitor                             | `MONITORS`                                                                                                                            |
| Visualization                       | `hook_name` and optional `hook_params` in visualization YAML                                                                          |

## Maintenance Rules

- When adding a new alias, update this file in the same change.
- When removing or renaming an alias, update templates and config examples at the same time.
- When a hook is experimental, prototype-only, or incomplete, keep that note in the catalog so operators do not mistake
  it for a fully productionized default.
