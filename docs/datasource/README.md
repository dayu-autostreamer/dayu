# Datasource Datasets and Manifests

This document mirrors the datasource runtime guide so datasource contracts are discoverable from the repository-level `docs/` index.

## Unified Dataset Layout

`http_video` and `rtsp_video` use the same dataset organization:

```text
<source-dir>/
  data/
    clips/
      000001.mp4
      000002.mp4
      000003.mp4
  http_video/
    manifest.json
  rtsp_video/
    manifest.json
```

Design goals:

- media files are stored only once under `data/`
- `http_video` and `rtsp_video` keep separate playback manifests
- manifests act as the transport-facing index layer
- `hash_data` stays aligned with ground-truth frame indices

This follows the common pattern used by mature video systems:

- compressed videos are the primary storage unit and decoded on demand
- manifests/index files are separated from the media payload
- playback order is explicitly defined by a playlist-like file

## Manifest Format

Each mode directory must contain a `manifest.json`.

Recommended schema:

```json
{
  "version": 1,
  "type": "video_sequence",
  "video_root": "../data",
  "sequence": [
    {
      "name": "road-segment-000001",
      "path": "clips/000001.mp4",
      "frame_count": 900,
      "start_frame_index": 0
    },
    {
      "name": "road-segment-000002",
      "path": "clips/000002.mp4",
      "frame_count": 900
    }
  ]
}
```

Field meanings:

- `version`: manifest schema version, currently `1`
- `type`: recommended value `video_sequence`
- `video_root`: path from the manifest directory to the shared media directory
- `sequence`: ordered clip list for this mode
- `sequence[].path`: clip path relative to `video_root`
- `sequence[].frame_count`: optional but strongly recommended to avoid startup probing
- `sequence[].start_frame_index`: optional global ground-truth start index for this clip

If `start_frame_index` is omitted, Dayu assigns it cumulatively from the previous clip. That is suitable when your ground-truth file is indexed over the same concatenated clip order starting at frame `0`.

## Frame Index and Hash Data

For `http_video`, the returned `hash_data` is the real frame index in ground-truth space.

More concretely:

- the first decoded frame of a clip maps to `start_frame_index`
- the second decoded frame maps to `start_frame_index + 1`
- and so on

This is the index consumed by `dependency/core/lib/estimation/accuracy_estimation.py`, where `search_frame_index()` directly treats `hash_data` as the ground-truth frame index.

That means:

- if your `gt_file.txt` is indexed from `0`, keep the first clip at `start_frame_index: 0`
- if your clips are extracted from a longer original recording, set each clip's `start_frame_index` to the corresponding original frame offset

## Mode Behavior

### `http_video`

- reads the ordered clip sequence from `http_video/manifest.json`
- decodes frames on demand from compressed video files
- samples frames according to the current generator policy
- returns sampled frame indices aligned with `gt_file.txt`

### `rtsp_video`

- reads the ordered clip sequence from `rtsp_video/manifest.json`
- streams the listed clips in manifest order to RTSP
- reuses the same underlying `data/` media files as `http_video`

## Recommended Practice

For best stability:

- keep all clips in a single codec/container family, ideally `.mp4`
- keep `frame_count` in the manifest so startup does not need probing
- set `start_frame_index` explicitly whenever ground-truth indexing is not a simple contiguous `0..N-1`
- if `http_video` and `rtsp_video` need the same clip order, reuse the same manifest content or create one as a symlink/copy of the other
