# Docker Images For Dayu

Dayu runs as a set of containerized cloud-edge services. The Dockerfiles in this
directory describe how each image is assembled; the build matrix itself lives in
[`../docker-bake.hcl`](../docker-bake.hcl).

This split keeps two concerns separate:

- `build/*.Dockerfile`: package one image, install its dependencies, and copy the
  required Dayu code.
- `docker-bake.hcl`: declare which images exist, which Dockerfile they use, which
  platforms they build for, and which tag variants they publish.

## Common Entry Points

Use `make` for normal development and CI-compatible builds:

```bash
make build WHAT=backend
make build WHAT=monitor,generator
make build WHAT=processors
make all
make validate-build
```

`make all` builds the same default image set as the historical `cross-build`
script: runtime images plus processor images. It does not build `dayubase` or
`rtsp-server` unless those targets are requested explicitly.

The build variables are:

| Variable | Default | Meaning |
| --- | --- | --- |
| `REG` / `REGISTRY` | `docker.io` | Output registry and default external base registry |
| `REPO` | `dayuhub` | Output repository/namespace |
| `TAG` / `IMAGE_TAG` | `v1.4` from `Makefile` | Output image tag |
| `BASE_REPO` | `dayuhub` | Repository/namespace for internal Dayu base images used in `FROM` |
| `BASE_TAG` | `latest` | Base `dayubase` tag used by JetPack-aware images |
| `NO_CACHE` / `NOCACHE` | `0` | Set to `1` or `true` to pass `--no-cache` |

Examples:

```bash
make build WHAT=traffic-signal-recognition TAG=v1.4 REG=repo:5000 REPO=dayuhub
make build WHAT=processors BASE_REPO=private-dayu BASE_TAG=v1.4
make all NOCACHE=1
```

To inspect the resolved Bake definition without building:

```bash
bash hack/make-rules/cross-build.sh --print --files monitor --tag v1.4
```

## BuildKit Configuration

`hack/resource/buildkitd.toml` and `hack/resource/driver_opts.toml` configure the
`dayu-buildx` builder. Their sibling `*_template.toml` files contain examples for
an HTTP registry and a proxied build host. Keep the output registry in
`NO_PROXY`; comma-separated values are passed to Buildx as one driver option.

Multi-platform processor builds run arm64 package installation through QEMU and
can exhaust a small build VM when BuildKit schedules too many steps at once. On
a host with about 10 GiB of memory, copy the limit from
`buildkitd_template.toml` into `buildkitd.toml`:

```toml
[worker.oci]
  max-parallelism = 2
```

Increase the value cautiously on larger hosts, or lower it if the kernel still
reports OOM kills during multi-platform builds.

BuildKit reads these files when the builder is created. After changing them,
recreate the builder while retaining its cache state, then rerun the build:

```bash
docker buildx rm --keep-state dayu-buildx
make all
```

An insecure HTTP registry belongs in a `[registry."host:port"]` block, as shown
in `buildkitd_template.toml`; registry addresses are not BuildKit entitlement
names.

## Bake Targets And Groups

The main groups in `docker-bake.hcl` are:

| Target/group | Purpose |
| --- | --- |
| `default` | Runtime images plus processor images, matching `make all` |
| `runtime` | Backend, frontend, datasource, generator, distributor, controller, scheduler, monitor |
| `processors` | All application service processor images |
| `rtsp-server` | RTSP server utility image |
| `dayubase` | Arch-specific dayubase images used before manifest creation |
| `all-images` | Default set plus `rtsp-server` and dayubase arch tags |

Most runtime images publish a single tag. `monitor` and processor images publish
four tags automatically:

- `TAG`
- `TAG-jp4`
- `TAG-jp5`
- `TAG-jp6`

These variants line up with the deployment logic in `backend/runtime_orchestrator.py`,
which uses the already-fetched node inventory to append `-jpX` to edge monitor and
processor images when a node reports a known JetPack major version. This selection
does not add worker-side Kubernetes discovery.

## Dayubase

`dayubase` is special because amd64 and Jetson arm64 variants use different
Dockerfiles. Build it through the dedicated wrapper so the arch images and final
manifest tags are both created:

```bash
bash hack/tools/build_dayubase.sh --tag latest --jp default
bash hack/tools/build_dayubase.sh --tag latest --jp all
```

The wrapper builds the relevant arch targets from `docker-bake.hcl`, then creates
the multi-arch manifest tag with `docker buildx imagetools create`.

## Consistency Checks

Run this after adding, renaming, or removing images:

```bash
make validate-build
```

The validator checks that:

- every `build/*.Dockerfile` is referenced by `docker-bake.hcl`
- every Dockerfile referenced by Bake exists
- every `template/**.yaml` `image:` reference has a matching Bake target
- Dayu component Dockerfiles use `BASE_REPO` instead of hard-coding internal
  Dayu base-image repositories
