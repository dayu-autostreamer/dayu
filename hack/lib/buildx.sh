#!/usr/bin/env bash
set -o errexit
set -o nounset
set -o pipefail

# ============================================================
# Build logic:
# - Normal components: build exactly ONCE with TAG.
# - Special components (defined by SPECIAL_TAG_IMAGES array in import_docker_info):
#   build 4 variants automatically: TAG, TAG-jp4, TAG-jp5, TAG-jp6
#   (each variant is multi-arch if PLATFORMS includes amd64,arm64)
#
# Special components Dockerfile typically contains:
#   ARG TAG=latest
#   FROM .../dayubase:${TAG}
# pass --build-arg TAG=<variant-tag> during build.
#
# Normal components Dockerfile typically contains:
#   FROM .../dayubase:latest
# do not need build-arg TAG.
# ============================================================


# -----------------------------
# Helpers
# -----------------------------
die() { echo "ERROR: $*" >&2; exit 1; }

dayu::buildx::read_driver_opts() {
  local driver_opts_file="$1"
  local -n _driver_opts_array="$2"

  _driver_opts_array=()
  if [[ -f "$driver_opts_file" ]]; then
    while IFS= read -r line; do
      [[ -z "$line" || "$line" =~ ^# ]] && continue
      if [[ "$line" =~ = ]]; then
        key=$(echo "$line" | awk -F'=' '{gsub(/^[ \t]+|[ \t]+$/, "", $1); print $1}')
        value=$(echo "$line" | awk -F'=' '{gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2}')
        value=$(echo "$value" | sed 's/^"\(.*\)"$/\1/')
        if [[ "$value" =~ , ]]; then
          _driver_opts_array+=( --driver-opt \"$key=$value\" )
        else
          _driver_opts_array+=( --driver-opt "$key=$value" )
        fi
      fi
    done < "$driver_opts_file"
  fi
  echo "driver opts in buildx creating: " "${_driver_opts_array[@]}"
}

dayu::buildx::prepare_env() {
  if ! docker buildx >/dev/null 2>&1; then
    echo "ERROR: docker buildx not available. Docker 19.03 or higher is required with experimental features enabled.
    Please refer to https://dayu-autostreamer.github.io/docs/developer-guide/how-to-build/build-preparation for buildx instructions." >&2
    exit 1
  fi

  docker run --privileged --rm tonistiigi/binfmt --install all

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

dayu::buildx::import_docker_info() {
  declare -g -A DOCKERFILES=(
      [backend]="build/backend.Dockerfile"
      [frontend]="build/frontend.Dockerfile"
      [datasource]="build/datasource.Dockerfile"

      [generator]="build/generator.Dockerfile"
      [distributor]="build/distributor.Dockerfile"
      [controller]="build/controller.Dockerfile"
      [monitor]="build/monitor.Dockerfile"
      [scheduler]="build/scheduler.Dockerfile"

      [car-detection]="build/car_detection.Dockerfile"
      [face-detection]="build/face_detection.Dockerfile"
      [gender-classification]="build/gender_classification.Dockerfile"
      [age-classification]="build/age_classification.Dockerfile"
      [model-switch-detection]="build/model_switch_detection.Dockerfile"
      [pedestrian-detection]="build/pedestrian_detection.Dockerfile"
      [license-plate-recognition]="build/license_plate_recognition.Dockerfile"
      [vehicle-detection]="build/vehicle_detection.Dockerfile"
      [exposure-identification]="build/exposure_identification.Dockerfile"
      [category-identification]="build/category_identification.Dockerfile"
  )

  declare -g -A PLATFORMS=(
      [backend]="linux/amd64"
      [frontend]="linux/amd64"
      [datasource]="linux/amd64,linux/arm64"

      [generator]="linux/amd64,linux/arm64"
      [distributor]="linux/amd64"
      [controller]="linux/amd64,linux/arm64"
      [monitor]="linux/amd64,linux/arm64"
      [scheduler]="linux/amd64"

      [car-detection]="linux/amd64,linux/arm64"
      [face-detection]="linux/amd64,linux/arm64"
      [gender-classification]="linux/amd64,linux/arm64"
      [age-classification]="linux/amd64,linux/arm64"
      [model-switch-detection]="linux/amd64,linux/arm64"
      [pedestrian-detection]="linux/amd64,linux/arm64"
      [license-plate-recognition]="linux/amd64,linux/arm64"
      [vehicle-detection]="linux/amd64,linux/arm64"
      [exposure-identification]="linux/amd64,linux/arm64"
      [category-identification]="linux/amd64,linux/arm64"
  )

  # ----------------------------------------------------------
  # Special images that must build 4 tags: TAG / TAG-jp4/5/6
  # ----------------------------------------------------------
  declare -g -a SPECIAL_TAG_IMAGES=(
    monitor
    car-detection
    face-detection
    gender-classification
    age-classification
    model-switch-detection
    pedestrian-detection
    license-plate-recognition
    vehicle-detection
    exposure-identification
    category-identification
  )
}

dayu::buildx::import_env_variables(){
  NO_CACHE=false
  BASE_TAG="latest"
  SELECTED_FILES=""

  # Parse command line arguments
  while [[ $# -gt 0 ]]; do
      case "$1" in
          --files)
              [[ -n "${2:-}" ]] || die '"--files" requires a non-empty option argument.'
              SELECTED_FILES=$2
              shift
              ;;
          --tag)
              [[ -n "${2:-}" ]] || die '"--tag" requires a non-empty option argument.'
              TAG=$2
              shift
              ;;
          --repo)
              [[ -n "${2:-}" ]] || die '"--repo" requires a non-empty option argument.'
              REPO=$2
              shift
              ;;
          --registry)
              [[ -n "${2:-}" ]] || die '"--registry" requires a non-empty option argument.'
              REGISTRY=$2
              shift
              ;;
          --no-cache)
              NO_CACHE=true
              ;;
          --) shift; break ;;
          *) break ;;
      esac
      shift
  done
}

dayu::buildx::init_env(){
  dayu::buildx::prepare_env
  dayu::buildx::import_docker_info
  dayu::buildx::import_env_variables "$@"
}

# ------------------------------------------------------------
# Special-image decision based on SPECIAL_TAG_IMAGES array
# ------------------------------------------------------------
dayu::buildx::is_special_image() {
  local image="$1"
  local x
  for x in "${SPECIAL_TAG_IMAGES[@]}"; do
    if [[ "$x" == "$image" ]]; then
      return 0
    fi
  done
  return 1
}

# ------------------------------------------------------------
# Build functions
# ------------------------------------------------------------
dayu::buildx::build_normal_image() {
  local image="$1"
  local platform="$2"
  local dockerfile="$3"
  local cache_option="$4"

  local image_tag="${REGISTRY}/${REPO}/${image}:${TAG}"
  local context_dir="."

  echo "Building NORMAL image: ${image_tag} platform: ${platform} dockerfile: ${dockerfile} no-cache: ${NO_CACHE}"

  if [[ -z "${cache_option}" ]]; then
    docker buildx build \
      --platform "${platform}" \
      --build-arg REG="${REGISTRY}" \
      -t "${image_tag}" \
      -f "${dockerfile}" \
      "${context_dir}" \
      --push
  else
    docker buildx build \
      --platform "${platform}" \
      --build-arg REG="${REGISTRY}" \
      -t "${image_tag}" \
      -f "${dockerfile}" \
      "${context_dir}" \
      "${cache_option}" \
      --push
  fi
}

dayu::buildx::build_special_arch() {
  local image="$1"
  local platform="$2"      # linux/amd64 or linux/arm64
  local dockerfile="$3"
  local variant_tag="$4"   # TAG / TAG-jp4 / TAG-jp5 / TAG-jp6
  local cache_option="$5"

  local arch="${platform##*/}"
  local temp_tag="${REGISTRY}/${REPO}/${image}:${variant_tag}-${arch}"
  local context_dir="."

  echo "Building SPECIAL arch image: ${temp_tag} platform: ${platform} dockerfile: ${dockerfile} baseTAG: ${variant_tag} no-cache: ${NO_CACHE}"

  if [[ -z "${cache_option}" ]]; then
    docker buildx build \
      --platform "${platform}" \
      --build-arg REG="${REGISTRY}" \
      --build-arg TAG="${variant_tag}" \
      -t "${temp_tag}" \
      -f "${dockerfile}" \
      "${context_dir}" \
      --push
  else
    docker buildx build \
      --platform "${platform}" \
      --build-arg REG="${REGISTRY}" \
      --build-arg TAG="${variant_tag}" \
      -t "${temp_tag}" \
      -f "${dockerfile}" \
      "${context_dir}" \
      "${cache_option}" \
      --push
  fi
}

dayu::buildx::create_and_push_manifest() {
  local image="$1"
  local variant_tag="$2"

  local manifest_tag="${REGISTRY}/${REPO}/${image}:${variant_tag}"
  echo "Creating and pushing manifest for: ${manifest_tag}"

  docker buildx imagetools create -t "${manifest_tag}" \
    "${REGISTRY}/${REPO}/${image}:${variant_tag}-amd64" \
    "${REGISTRY}/${REPO}/${image}:${variant_tag}-arm64"
}

dayu::buildx::build_special_image_all_variants() {
  local image="$1"
  local platform_csv="$2"
  local dockerfile="$3"
  local cache_option="$4"

  local -a variants=("${BASE_TAG}" "${BASE_TAG}-jp4" "${BASE_TAG}-jp5" "${BASE_TAG}-jp6")

  if [[ "${platform_csv}" != *","* ]]; then
    # Single-platform special image: build final tag directly (no manifest)
    for vt in "${variants[@]}"; do
      local image_tag="${REGISTRY}/${REPO}/${image}:${vt}"
      echo "Building SPECIAL single-platform image: ${image_tag} platform: ${platform_csv} dockerfile: ${dockerfile} baseTAG: ${vt}"

      if [[ -z "${cache_option}" ]]; then
        docker buildx build \
          --platform "${platform_csv}" \
          --build-arg REG="${REGISTRY}" \
          --build-arg TAG="${vt}" \
          -t "${image_tag}" \
          -f "${dockerfile}" \
          "." \
          --push
      else
        docker buildx build \
          --platform "${platform_csv}" \
          --build-arg REG="${REGISTRY}" \
          --build-arg TAG="${vt}" \
          -t "${image_tag}" \
          -f "${dockerfile}" \
          "." \
          "${cache_option}" \
          --push
      fi
    done
    return 0
  fi

  # Multi-arch special image: build per-arch temp tags then manifest per variant
  IFS=',' read -ra plats <<< "${platform_csv}"
  for vt in "${variants[@]}"; do
    for p in "${plats[@]}"; do
      p="$(echo "$p" | xargs)"
      dayu::buildx::build_special_arch "${image}" "${p}" "${dockerfile}" "${vt}" "${cache_option}"
    done
    dayu::buildx::create_and_push_manifest "${image}" "${vt}"
  done
}

# ------------------------------------------------------------
# Main entry
# ------------------------------------------------------------
dayu::buildx::build_and_push_multi_platform_images(){
  dayu::buildx::init_env "$@"

  local CACHE_OPTION=""
  if [[ "${NO_CACHE}" = true ]]; then
    CACHE_OPTION="--no-cache"
  fi

  local -a targets=()
  if [[ -n "${SELECTED_FILES}" ]]; then
    IFS=',' read -ra targets <<< "${SELECTED_FILES}"
  else
    echo "No images specified, building all default images."
    for k in "${!DOCKERFILES[@]}"; do targets+=("$k"); done
  fi

  echo "Registry : ${REGISTRY}"
  echo "Repo     : ${REPO}"
  echo "Tag      : ${TAG}"
  echo "No-cache : ${NO_CACHE}"
  echo "Targets  : ${targets[*]}"
  echo "Special  : ${SPECIAL_TAG_IMAGES[*]}"
  echo ""

  for image in "${targets[@]}"; do
    if [[ -z "${DOCKERFILES[$image]:-}" || -z "${PLATFORMS[$image]:-}" ]]; then
      echo "Unknown image or platform not specified: ${image} (skipped)"
      continue
    fi

    local dockerfile="${DOCKERFILES[$image]}"
    local platform="${PLATFORMS[$image]}"

    if dayu::buildx::is_special_image "${image}"; then
      echo "------------------------------------------------------------"
      echo "SPECIAL build: ${image}  (will build: ${TAG}, ${TAG}-jp4, ${TAG}-jp5, ${TAG}-jp6)"
      echo "platforms: ${platform}"
      echo "dockerfile: ${dockerfile}"
      echo "------------------------------------------------------------"
      dayu::buildx::build_special_image_all_variants "${image}" "${platform}" "${dockerfile}" "${CACHE_OPTION}"
    else
      echo "------------------------------------------------------------"
      echo "NORMAL build: ${image} (will build: ${TAG})"
      echo "platforms: ${platform}"
      echo "dockerfile: ${dockerfile}"
      echo "------------------------------------------------------------"
      dayu::buildx::build_normal_image "${image}" "${platform}" "${dockerfile}" "${CACHE_OPTION}"
    fi
  done

  echo ""
  echo "Done."
}

dayu::buildx::build_and_push_multi_platform_images "$@"
