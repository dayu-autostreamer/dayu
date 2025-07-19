#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

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
  # Check whether buildx exists.
  if ! docker buildx >/dev/null 2>&1; then
    echo "ERROR: docker buildx not available. Docker 19.03 or higher is required with experimental features enabled.
    Please refer to https://dayu-autostreamer.github.io/docs/developer-guide/how-to-build/build-preparation for buildx instructions." >&2
    exit 1
  fi

  # Use tonistiigi/binfmt that is to enable an execution of different multi-architecture containers
  docker run --privileged --rm tonistiigi/binfmt --install all

  # Create a new builder which gives access to the new multi-architecture features.
  local BUILDER_INSTANCE="dayu-buildx"
  local BUILDKIT_CONFIG_FILE="${DAYU_ROOT}/hack/resource/buildkitd.toml"
  local DRIVER_OPTS_FILE="${DAYU_ROOT}/hack/resource/driver_opts.toml"

  if ! docker buildx inspect $BUILDER_INSTANCE >/dev/null 2>&1; then
    local -a DRIVER_OPTS=()
    dayu::buildx::read_driver_opts "$DRIVER_OPTS_FILE" DRIVER_OPTS
     docker buildx create \
      --use \
      --name "$BUILDER_INSTANCE" \
      --driver docker-container \
      --config "$BUILDKIT_CONFIG_FILE" \
      "${DRIVER_OPTS[@]}"
  fi
  docker buildx use $BUILDER_INSTANCE
}

dayu::buildx::import_docker_info() {
  # Default Dockerfiles and their platforms
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
  )
  # Corresponding platforms
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
  )
  # Images requiring special treatment, their platforms, and Dockerfiles
  declare -g -A SPECIAL_BUILD=(
      [face-detection]="linux/amd64:build/face_detection_amd64.Dockerfile,linux/arm64:build/face_detection_arm64.Dockerfile"
      [gender-classification]="linux/amd64:build/gender_classification_amd64.Dockerfile,linux/arm64:build/gender_classification_arm64.Dockerfile"
      [age-classification]="linux/amd64:build/age_classification_amd64.Dockerfile,linux/arm64:build/age_classification_arm64.Dockerfile"
      [model-switch-detection]="linux/amd64:build/model_switch_detection_amd64.Dockerfile,linux/arm64:build/model_switch_detection_arm64.Dockerfile"
  )
}


dayu::buildx::import_env_variables(){
  NO_CACHE=false
  SELECTED_FILES=""
  # Parse command line arguments
  while [[ $# -gt 0 ]]; do
      case "$1" in
          --files)
              if [ "$2" ]; then
                  SELECTED_FILES=$2
                  shift
              else
                  echo 'ERROR: "--files" requires a non-empty option argument.'
                  exit 1
              fi
              ;;
          --tag)
              if [ "$2" ]; then
                  TAG=$2
                  shift
              else
                  echo 'ERROR: "--tag" requires a non-empty option argument.'
                  exit 1
              fi
              ;;
          --repo)
              if [ "$2" ]; then
                  REPO=$2
                  shift
              else
                  echo 'ERROR: "--repo" requires a non-empty option argument.'
                  exit 1
              fi
              ;;
          --registry)
              if [ "$2" ]; then
                  REGISTRY=$2
                  shift
              else
                  echo 'ERROR: "--repo" requires a non-empty option argument.'
                  exit 1
              fi
              ;;
          --no-cache)
              NO_CACHE=true
              ;;
          --)              # End of options
              shift
              break
              ;;
          *)               # Default case
              break
      esac
      shift
  done
}

dayu::buildx::init_env(){
  dayu::buildx::prepare_env
  dayu::buildx::import_docker_info
  dayu::buildx::import_env_variables "$@"
}

dayu::buildx::build_image() {
    local image=$1
    local platform=$2
    local dockerfile=$3
    local cache_option=$4  # --no-cache or empty
    local image_tag="${REGISTRY}/${REPO}/${image}:${TAG}"
    local context_dir="."

    echo "Building image: $image_tag on platform: $platform using Dockerfile: $dockerfile with no-cache: $NO_CACHE"

    if [ -z "$cache_option" ]; then
        docker buildx build --platform "$platform" --build-arg REG="${REGISTRY}" -t "$image_tag" -f "$dockerfile" "$context_dir" --push
    else
        docker buildx build  --platform "$platform" --build-arg REG="${REGISTRY}" -t "$image_tag" -f "$dockerfile" "$context_dir" "$cache_option" --push
    fi
}

dayu::buildx::build_image_special() {
    local image=$1
    local platform=$2
    local dockerfile=$3
    local cache_option=$4  # --no-cache or empty
    local temp_tag="${REGISTRY}/${REPO}/${image}:${TAG}-${platform##*/}"  # Temporary tag for the build
    local context_dir="."

    echo "Building image: $temp_tag on platform: $platform using Dockerfile: $dockerfile with no-cache: $NO_CACHE"

    if [ -z "$cache_option" ]; then
         docker  buildx build --platform="$platform" --build-arg REG="${REGISTRY}" -t "$temp_tag" -f "$dockerfile" "$context_dir" --push
    else
         docker  buildx build  --platform="$platform" --build-arg REG="${REGISTRY}" -t "$temp_tag" -f "$dockerfile" "$context_dir" "$cache_option" --push
    fi
}

dayu::buildx::create_and_push_manifest() {
    local image=$1
    local tag=$2
    local repo=$3
    local manifest_tag="${REGISTRY}/${repo}/${image}:${tag}"

    echo "Creating and pushing manifest for: $manifest_tag"

    docker buildx imagetools create -t "$manifest_tag" \
        "${REGISTRY}/${repo}/${image}:${tag}-amd64" \
        "${REGISTRY}/${repo}/${image}:${tag}-arm64"

}

dayu::buildx::build_and_push_multi_platform_images(){
  dayu::buildx::init_env "$@"

  # Determine if --no-cache should be used
  CACHE_OPTION=""
  if [ "$NO_CACHE" = true ] ; then
      CACHE_OPTION="--no-cache"
  fi

  # specified platforms
  if [ -n "$SELECTED_FILES" ]; then
      IFS=',' read -ra ADDR <<< "$SELECTED_FILES"
      for image in "${ADDR[@]}"; do
          if [[ -n "${DOCKERFILES[$image]}" && -n "${PLATFORMS[$image]}" ]]; then
              # Check if it's a specially treated image
              if [[ -n "${SPECIAL_BUILD[$image]:-}" ]]; then
                  IFS=',' read -ra SPECIAL_PLATFORMS <<< "${SPECIAL_BUILD[$image]}"
                  for entry in "${SPECIAL_PLATFORMS[@]}"; do
                      IFS=':' read -ra DETAILS <<< "$entry"
                      platform="${DETAILS[0]}"
                      dockerfile="${DETAILS[1]}"
                      dayu::buildx::build_image_special "$image" "$platform" "$dockerfile" "$CACHE_OPTION"
                  done
                  # After building all architectures, create and push manifest
                  dayu::buildx::create_and_push_manifest "$image" "$TAG" "$REPO"
              else
                  dayu::buildx::build_image "$image" "${PLATFORMS[$image]}" "${DOCKERFILES[$image]}" "$CACHE_OPTION"
              fi
          else
              echo "Unknown image or platform not specified: $image"
          fi
      done
  else
      echo "No images specified, building all default images."
      for image in "${!DOCKERFILES[@]}"; do
          if [[ -n "${SPECIAL_BUILD[$image]:-}" ]]; then
              IFS=',' read -ra SPECIAL_PLATFORMS <<< "${SPECIAL_BUILD[$image]}"
              for entry in "${SPECIAL_PLATFORMS[@]}"; do
                  IFS=':' read -ra DETAILS <<< "$entry"
                  platform="${DETAILS[0]}"
                  dockerfile="${DETAILS[1]}"
                  dayu::buildx::build_image_special "$image" "$platform" "$dockerfile" "$CACHE_OPTION"
              done
              # After building all architectures, create and push manifest
              dayu::buildx::create_and_push_manifest "$image" "$TAG" "$REPO"
          else
              dayu::buildx::build_image "$image" "${PLATFORMS[$image]}" "${DOCKERFILES[$image]}" "$CACHE_OPTION"
          fi
      done
  fi

}
