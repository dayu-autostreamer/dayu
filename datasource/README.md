# Datasource for Dayu System

## HTTP Video Storage

`http_video` now supports compressed video files as the primary storage format.
The datasource keeps the external HTTP contract unchanged:

- `/source` still returns the sampled frame index list
- `/file` still returns the task video file consumed by generators/processors

This removes the need to pre-split every source video into `frames/*.jpg`, which
reduces storage pressure and makes it easier to add new datasets.

### Recommended layout

Store the source data directly under each `http_video/` directory:

```text
<camera-dir>/
  http_video/
    videos/
      0001.mp4
      0002.mp4
      0003.mp4
```

The datasource will recursively discover supported video files and play them in
natural filename order (`2.mp4` before `10.mp4`).

### Optional manifest

If you need a custom play order, add `manifest.json` (or
`datasource_manifest.json`) under `http_video/`:

```json
{
  "files": [
    "videos/clip_b.mp4",
    "videos/clip_a.mp4"
  ]
}
```

Each entry can be either:

- a relative file path string
- an object with a `path` field

Example:

```json
{
  "files": [
    {"path": "videos/morning.mp4"},
    {"path": "videos/noon.mp4"}
  ]
}
```

### Legacy layout

The old frame directory is still supported for backward compatibility:

```text
<camera-dir>/
  http_video/
    frames/
      0.jpg
      1.jpg
      2.jpg
```

If `frames/` exists, the datasource will keep using the legacy reader.

### Supported video formats

Common containers such as `.mp4`, `.avi`, `.mov`, `.mkv`, `.m4v`, `.ts`,
`.webm`, and `.flv` are supported.


