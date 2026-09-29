# STEADY and UNSTEADY

STEADY combines background macro search with per-request micro feedback. `unsteady-macro` uses only the macro search
result; `unsteady-micro` uses fixed feedback weights. Both select the `unsteady` agent with `schedule_type` set to
`macro` or `micro`. They share STEADY's prediction, clustering, task parsing and deployment validation.

## Runtime contract

These research policies support a **linear detection pipeline**, optionally followed by one classification service.
The execution model has one detection stage and one optional per-object classification stage. Branches, joins,
additional stages, topology changes and source-device changes are rejected. This does not restrict other Dayu policies.

The agent selects FPS, resolution, buffer size and an edge-to-cloud split. It uses the shared pipeline helpers to
preserve node metadata, place `_start` on the source edge and `_end` on cloud, and validate every target against the
LIVE RuntimeDirectory. The templates use `source-edge-cloud` initial deployment and `non` redeployment. They do not
perform Kubernetes discovery or change the framework's task/routing contracts.

Before installing one of these policies:

- Provide offline profiles under the scheduler volume's `test_kb/kb2` directory, or set `kb_path` to another mounted
  directory. Profiles are workload/hardware measurements and are not bundled with the repository.
- Enable `['obj_num', 'obj_size', 'obj_velocity']` in the selected detector template's `SCENARIOS_EXTRACTORS`.
  The default detector templates omit optical flow to avoid its overhead for unrelated policies. Until both scenario
  features and bandwidth observations arrive, the scheduler keeps the configured default policy.
- Provide FPS/resolution/buffer ranges that include `default_policy`; buffer sizes must be at least two frames.
  Accuracy sampling retains the algorithm's 30 FPS, 1080p, two-frame all-cloud reference task, so the source and
  profiles must support those settings as well.

The task result contract is `outputs.bbox[].items[].bbox`. Accuracy feedback compares the previous task's last frame
with the reference task's first frame, with coordinates scaled to the reference resolution. Missing frame records
are skipped rather than treated as a measured accuracy. Object velocity reads the task's revision-scoped temporary
file, releases the video handle and tolerates empty/untrackable regions.

## Offline profile schema

Each business service has a `<service-name>.json` execution profile. Prefer role-based keys:

```json
{
  "execute_role=edge#resolution=360p": 0.04,
  "execute_role=cloud#resolution=360p": 0.01
}
```

These example values are seconds per frame for detection and seconds per object for classification. Measure every
resolution and role used by the search. Existing `execute_device` profiles are accepted only when the device label
unambiguously identifies an edge or cloud role; ambiguous or duplicate entries fail instead of guessing a hostname.

`file_size.json` maps `resolution=360p#fps=30#encoding=mp4v#buffer_size=4` to the measured video size used by the
existing transmission model. Keep the same size/bandwidth units as the profiling setup. Cover every combination of
resolution, FPS and buffer size in the search range, including accuracy samples.

## Parameters

Pass `sch_param` plus `steady_param` (STEADY) or `unsteady_param` (UNSTEADY) in `SCH_AGENT_PARAMETERS`.

| Parameter | Meaning |
| --- | --- |
| `delay_cons`, `acc_cons`, `delay_weight` | Single-run constraints; accuracy weight is `1 - delay_weight`. |
| `delay_cons_adjust`, `acc_cons_adjust`, `delay_weight_adjust` | Optional multipliers applied to scheduler constraints, not raw values written in context records. |
| `history_length`, `context_analyze_type` | Context history length; `1` averages samples, `0` uses conservative extrema. Existing `history_lenghth` and `context_anylze_type` spellings remain readable. |
| `cluster_threshold` | Context-bin acceptance fraction in `[0, 1]`; `0` disables clustering. Bin boundaries and conservative representatives retain the research model. |
| `corrector_param.corrector_pool_threshold` | Relative-error threshold for invalidating pooled correctors; defaults to `0.2`. |
| `if_online_train` | Enable per-context classifier training. New templates enable it for macro search and disable it for micro-only feedback. Missing classifiers fall back to greedy search. |
| `macro_update_interval`, `stop_threshold` | Background search interval and relative stopping threshold. |
| `coeff_info.step_coeff` | Initial, additive-increase, minimum and maximum feedback step coefficients. |
| `acc_sample_interval` | Positive number of scheduling requests between reference tasks. |
| `feedback_weight` | For micro-only mode, `delay_decrease_weight` and `acc_increase_weight` each cover all four knobs. |

The micro-only template supplies normalized opposing FPS/resolution directions and zero buffer/placement weights.
These are usable starting values, **not reproduced experiment settings**. Set measured weights appropriate to the
workload and the ordering of the configured knob ranges before comparing research results. The feedback formula,
step adaptation and clipping are shared with STEADY.

For controlled constraint sweeps, supply all or any of `all_delay_cons_info`, `all_acc_cons_info` and
`all_delay_weight_info` as nonempty lists of `{'value': ..., 'adjust': ...}` entries. A supplied list overrides its
corresponding scalar. The Cartesian-product order is weight, delay, accuracy. Enable
`sch_param.if_stop_record_in_single_cycle`; each combination consumes `sch_param.stop_max_frame_num` logical source
frames, or `unit_logic_frame_num_max` when supplied in the algorithm parameters. After the final combination,
recording stops while scheduling continues with the last constraints. No generation admission or deployment changes
are introduced by a sweep.

Record paths are optional. The STEADY template writes context, scheduling and prediction JSON lines under `records/`
in the mounted scheduler directory. UNSTEADY records task/context data. Old context records without `cons_table`
remain readable. Each agent owns its search/training workers and exposes `stop()` to release them during teardown.

The shared predictor also retains the four-argument constructor used by AdaMEC, Gecko and MadEye. Those consumers
have clustering disabled unless they explicitly supply a threshold; their policy-search implementations are unchanged.
