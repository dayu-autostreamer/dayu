#!/usr/bin/env bash
set -o errexit
set -o nounset
set -o pipefail

die() {
  echo "ERROR: $*" >&2
  exit 1
}

dayu::buildx::read_driver_opts() {
  local driver_opts_file="$1"
  local -n _driver_opts_array="$2"

  _driver_opts_array=()
  if [[ -f "$driver_opts_file" ]]; then
    local line key value driver_opt
    while IFS= read -r line; do
      [[ -z "$line" || "$line" =~ ^# ]] && continue
      if [[ "$line" =~ = ]]; then
        key=$(echo "$line" | awk -F'=' '{gsub(/^[ \t]+|[ \t]+$/, "", $1); print $1}')
        value=$(echo "$line" | awk -F'=' '{gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2}')
        value=$(echo "$value" | sed 's/^"\(.*\)"$/\1/')
        driver_opt="$key=$value"
        # Buildx parses driver options as CSV. Keep comma-containing values,
        # such as NO_PROXY, together as one quoted CSV field.
        if [[ "$driver_opt" == *,* ]]; then
          driver_opt="\"$driver_opt\""
        fi
        _driver_opts_array+=(--driver-opt "$driver_opt")
      fi
    done < "$driver_opts_file"
  fi
  echo "driver opts in buildx creating: " "${_driver_opts_array[@]}"
}

dayu::buildx::prepare_env() {
  if ! docker buildx >/dev/null 2>&1; then
    echo "ERROR: docker buildx not available. Docker 19.03 or higher is required with experimental features enabled.
    Please refer to https://dayu-autostreamer.github.io/docs/developer-guide/install-docker-buildx for buildx instructions." >&2
    exit 1
  fi

  if [[ "${DAYU_BUILDX_SKIP_BINFMT:-false}" != "true" ]]; then
    docker run --privileged --rm tonistiigi/binfmt --install all
  fi

  local BUILDER_INSTANCE="dayu-buildx"
  local BUILDKIT_CONFIG_FILE="${DAYU_ROOT}/hack/resource/buildkitd.toml"
  local DRIVER_OPTS_FILE="${DAYU_ROOT}/hack/resource/driver_opts.toml"

  if ! docker buildx inspect "$BUILDER_INSTANCE" >/dev/null 2>&1; then
    local -a DRIVER_OPTS=()
    dayu::buildx::read_driver_opts "$DRIVER_OPTS_FILE" DRIVER_OPTS
    docker buildx create \
      --use \
      --name "$BUILDER_INSTANCE" \
      --driver docker-container \
      --config "$BUILDKIT_CONFIG_FILE" \
      "${DRIVER_OPTS[@]}"
  fi
  docker buildx use "$BUILDER_INSTANCE"
}

dayu::buildx::show_help() {
  cat << EOF
Usage: cross-build.sh [--files TARGETS] [--tag TAG] [--repo REPO] [--registry REG] [--base-repo REPO]
                      [--base-tag TAG] [--no-cache] [--print] [--help]

--files       Comma-separated Bake targets or groups. Default: default.
              Examples: backend,monitor,traffic-signal-recognition,processors,rtsp-server
--tag         Output image tag. Default: latest.
--repo        Output image repository/namespace. Default: dayuhub.
--registry    Output image registry. Default: \${REG:-docker.io}.
--base-repo   Repository/namespace for internal Dayu base images. Default: dayuhub.
--base-tag    Base dayubase tag used by special JP images. Default: latest.
--no-cache    Disable Docker build cache for selected targets.
--print       Print the resolved Bake definition instead of building.
--help        Display this help message and exit.

The build matrix lives in docker-bake.hcl. This wrapper only prepares buildx and
translates the historical Dayu command-line flags into Bake variables.
EOF
}

dayu::buildx::trim_target() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

dayu::buildx::parse_args() {
  NO_CACHE=false
  PRINT_ONLY=false
  SELECTED_FILES=""
  TAG="${TAG:-latest}"
  REPO="${REPO:-dayuhub}"
  REGISTRY="${REGISTRY:-${REG:-docker.io}}"
  BASE_REPO="${BASE_REPO:-dayuhub}"
  BASE_TAG="${BASE_TAG:-latest}"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --help)
        dayu::buildx::show_help
        exit 0
        ;;
      --files)
        [[ -n "${2:-}" ]] || die '"--files" requires a non-empty option argument.'
        SELECTED_FILES="$2"
        shift
        ;;
      --tag)
        [[ -n "${2:-}" ]] || die '"--tag" requires a non-empty option argument.'
        TAG="$2"
        shift
        ;;
      --repo)
        [[ -n "${2:-}" ]] || die '"--repo" requires a non-empty option argument.'
        REPO="$2"
        shift
        ;;
      --registry)
        [[ -n "${2:-}" ]] || die '"--registry" requires a non-empty option argument.'
        REGISTRY="$2"
        shift
        ;;
      --base-repo)
        [[ -n "${2:-}" ]] || die '"--base-repo" requires a non-empty option argument.'
        BASE_REPO="$2"
        shift
        ;;
      --base-tag)
        [[ -n "${2:-}" ]] || die '"--base-tag" requires a non-empty option argument.'
        BASE_TAG="$2"
        shift
        ;;
      --no-cache)
        NO_CACHE=true
        ;;
      --print)
        PRINT_ONLY=true
        ;;
      --)
        shift
        break
        ;;
      *)
        die "Unknown build option: $1"
        ;;
    esac
    shift
  done
}

dayu::buildx::resolve_targets() {
  RESOLVED_TARGETS=()
  if [[ -n "${SELECTED_FILES}" ]]; then
    local -a raw_targets=()
    local raw target
    IFS=',' read -ra raw_targets <<< "${SELECTED_FILES}"
    for raw in "${raw_targets[@]}"; do
      target="$(dayu::buildx::trim_target "$raw")"
      [[ -n "$target" ]] || continue
      RESOLVED_TARGETS+=("$target")
    done
  else
    RESOLVED_TARGETS=("default")
  fi

  [[ ${#RESOLVED_TARGETS[@]} -gt 0 ]] || die "No build targets resolved."
}

dayu::buildx::run_bake() {
  local -a targets=("$@")
  local bake_file="${DAYU_ROOT}/docker-bake.hcl"
  [[ -f "$bake_file" ]] || die "Bake file not found: $bake_file"

  export REGISTRY REPO TAG BASE_REPO BASE_TAG

  local -a bake_args=(-f "$bake_file")
  if [[ "${NO_CACHE}" == "true" ]]; then
    bake_args+=(--set=*.no-cache=true)
  fi
  if [[ "${PRINT_ONLY}" == "true" ]]; then
    bake_args+=(--print)
  fi
  bake_args+=("${targets[@]}")

  docker buildx bake "${bake_args[@]}"
}

dayu::buildx::build_and_push_multi_platform_images() {
  dayu::buildx::parse_args "$@"
  dayu::buildx::resolve_targets

  echo "Registry : ${REGISTRY}"
  echo "Repo     : ${REPO}"
  echo "Tag      : ${TAG}"
  echo "Base repo: ${BASE_REPO}"
  echo "Base tag : ${BASE_TAG}"
  echo "No-cache : ${NO_CACHE}"
  echo "Targets  : ${RESOLVED_TARGETS[*]}"
  echo ""

  if [[ "${PRINT_ONLY}" != "true" ]]; then
    dayu::buildx::prepare_env
  fi
  dayu::buildx::run_bake "${RESOLVED_TARGETS[@]}"
}
