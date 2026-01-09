#!/bin/bash
set -euo pipefail

# -----------------------------
# Help
# -----------------------------
show_help() {
cat << EOF
Usage: ${0##*/} [--files [dayubase...]] [--jp JP] [--tag TAG] [--repo REPO] [--registry REG] [--no-cache] [--help]

--files        Specify the images to build, separated by commas. Options include:
               dayubase
               Default is to select all (currently only dayubase).

--jp           Select JetPack variant(s) for ARM64 Dockerfile:
               - default        (use dayubase_arm64.Dockerfile)  [DEFAULT]
               - jp4 / 4
               - jp5 / 5
               - jp6 / 6
               - all            (build 4 tags: default + jp4 + jp5 + jp6)
               You can pass comma-separated list, e.g.:
               --jp jp4,jp6

               Tag rule:
               - default -> TAG (default: latest)
               - jpX     -> TAG-jpX  (e.g., latest-jp5)

--tag          Base tag. Default is "latest".
               For jp4/jp5/jp6 builds, final tag becomes "\$TAG-jpX".

--repo         Repository/namespace. Default is "dayuhub".
--registry     Registry. Default is "\${REG:-docker.io}".
--no-cache     Build without cache.
--help         Display this help message and exit.

Notes:
- Each final image tag will contain BOTH amd64 and arm64 via a manifest list.
- amd64 always uses: build/dayubase_amd64.Dockerfile
- arm64 uses one of:
    build/dayubase_arm64.Dockerfile (default)
    build/dayubase_jp4.Dockerfile
    build/dayubase_jp5.Dockerfile
    build/dayubase_jp6.Dockerfile
EOF
}

# -----------------------------
# Config
# -----------------------------
# Images list (keep extensible)
declare -A AMD64_DOCKERFILE=(
  [dayubase]="build/dayubase_amd64.Dockerfile"
)

# ARM64 Dockerfiles by variant
declare -A ARM64_VARIANT_DOCKERFILE=(
  [default]="build/dayubase_arm64.Dockerfile"
  [jp4]="build/dayubase_jp4.Dockerfile"
  [jp5]="build/dayubase_jp5.Dockerfile"
  [jp6]="build/dayubase_jp6.Dockerfile"
)

PLATFORM_AMD64="linux/amd64"
PLATFORM_ARM64="linux/arm64"

# Defaults
SELECTED_FILES=""
JP_VARIANTS_RAW=""     # if empty -> default
TAG="latest"
REPO="dayuhub"
NO_CACHE=false
REGISTRY="${REG:-docker.io}"

# -----------------------------
# Arg parsing
# -----------------------------
while :; do
  case "${1:-}" in
    --help)
      show_help
      exit 0
      ;;
    --files)
      if [ -n "${2:-}" ]; then
        SELECTED_FILES="$2"
        shift
      else
        echo 'ERROR: "--files" requires a non-empty option argument.'
        exit 1
      fi
      ;;
    --jp)
      if [ -n "${2:-}" ]; then
        JP_VARIANTS_RAW="$2"
        shift
      else
        echo 'ERROR: "--jp" requires a non-empty option argument.'
        exit 1
      fi
      ;;
    --tag)
      if [ -n "${2:-}" ]; then
        TAG="$2"
        shift
      else
        echo 'ERROR: "--tag" requires a non-empty option argument.'
        exit 1
      fi
      ;;
    --repo)
      if [ -n "${2:-}" ]; then
        REPO="$2"
        shift
      else
        echo 'ERROR: "--repo" requires a non-empty option argument.'
        exit 1
      fi
      ;;
    --registry)
      if [ -n "${2:-}" ]; then
        REGISTRY="$2"
        shift
      else
        echo 'ERROR: "--registry" requires a non-empty option argument.'
        exit 1
      fi
      ;;
    --no-cache)
      NO_CACHE=true
      ;;
    --)
      shift
      break
      ;;
    "")
      break
      ;;
    *)
      # Unknown arg -> stop parsing (or error if you prefer)
      break
      ;;
  esac
  shift
done

# -----------------------------
# Helpers
# -----------------------------
die() {
  echo "ERROR: $*" >&2
  exit 1
}

normalize_variant() {
  # input: token -> output: default/jp4/jp5/jp6/all
  local v="$1"
  v="$(echo "$v" | tr '[:upper:]' '[:lower:]')"
  case "$v" in
    default) echo "default" ;;
    all)     echo "all" ;;
    jp4|4)   echo "jp4" ;;
    jp5|5)   echo "jp5" ;;
    jp6|6)   echo "jp6" ;;
    *) die "Unknown --jp variant: '$1' (allowed: default, jp4/jp5/jp6, 4/5/6, all)" ;;
  esac
}

# Build one arch image and push with temp arch-suffixed tag
build_one_arch() {
  local image="$1"         # e.g. dayubase
  local final_tag="$2"     # e.g. latest-jp5
  local arch="$3"          # amd64|arm64
  local platform="$4"      # linux/amd64|linux/arm64
  local dockerfile="$5"    # path

  local temp_tag="${REGISTRY}/${REPO}/${image}:${final_tag}-${arch}"
  local context_dir="."
  local cache_option=""

  if [ "$NO_CACHE" = true ]; then
    cache_option="--no-cache"
  fi

  echo "==> Building: ${temp_tag}"
  echo "    platform : ${platform}"
  echo "    dockerfile: ${dockerfile}"
  echo "    no-cache : ${NO_CACHE}"

  docker buildx build \
    --platform "${platform}" \
    --build-arg REG="${REGISTRY}" \
    -t "${temp_tag}" \
    -f "${dockerfile}" \
    "${context_dir}" \
    ${cache_option} \
    --push
}

# Create and push manifest that points to the arch tags
create_and_push_manifest() {
  local image="$1"
  local final_tag="$2"

  local manifest_tag="${REGISTRY}/${REPO}/${image}:${final_tag}"
  echo "==> Creating & pushing manifest: ${manifest_tag}"

  docker buildx imagetools create -t "${manifest_tag}" \
    "${REGISTRY}/${REPO}/${image}:${final_tag}-amd64" \
    "${REGISTRY}/${REPO}/${image}:${final_tag}-arm64"
}

# Build amd64+arm64 and then create a multi-arch manifest
build_multiarch_image_for_variant() {
  local image="$1"          # dayubase
  local variant="$2"        # default|jp4|jp5|jp6
  local arm64_df="$3"       # resolved arm64 dockerfile

  local amd64_df="${AMD64_DOCKERFILE[$image]:-}"
  [ -n "${amd64_df}" ] || die "No amd64 Dockerfile configured for image '${image}'"
  [ -f "${amd64_df}" ] || die "amd64 Dockerfile not found: ${amd64_df}"
  [ -f "${arm64_df}" ] || die "arm64 Dockerfile not found: ${arm64_df}"

  local final_tag=""
  if [ "${variant}" = "default" ]; then
    final_tag="${TAG}"
  else
    # variant like jp5 -> tag suffix -jp5
    final_tag="${TAG}-${variant}"
  fi

  # Build both arches
  build_one_arch "${image}" "${final_tag}" "amd64" "${PLATFORM_AMD64}" "${amd64_df}"
  build_one_arch "${image}" "${final_tag}" "arm64" "${PLATFORM_ARM64}" "${arm64_df}"

  # Manifest
  create_and_push_manifest "${image}" "${final_tag}"
}

# -----------------------------
# Determine selected images
# -----------------------------
images_to_build=()
if [ -n "${SELECTED_FILES}" ]; then
  IFS=',' read -ra images_to_build <<< "${SELECTED_FILES}"
else
  # default build all known images
  images_to_build=("${!AMD64_DOCKERFILE[@]}")
fi

# -----------------------------
# Determine JP variants list
# -----------------------------
variants_to_build=()
if [ -z "${JP_VARIANTS_RAW}" ]; then
  variants_to_build=("default")
else
  IFS=',' read -ra raw_list <<< "${JP_VARIANTS_RAW}"
  # if any token is all -> expand to 4 variants
  expanded=false
  for tok in "${raw_list[@]}"; do
    nv="$(normalize_variant "${tok}")"
    if [ "${nv}" = "all" ]; then
      variants_to_build=("default" "jp4" "jp5" "jp6")
      expanded=true
      break
    fi
  done

  if [ "${expanded}" = false ]; then
    # normalize each and dedup
    declare -A seen=()
    for tok in "${raw_list[@]}"; do
      nv="$(normalize_variant "${tok}")"
      # ignore "all" here because handled above
      if [ -z "${seen[$nv]+x}" ]; then
        variants_to_build+=("${nv}")
        seen[$nv]=1
      fi
    done
  fi
fi

# -----------------------------
# Build
# -----------------------------
echo "Registry : ${REGISTRY}"
echo "Repo     : ${REPO}"
echo "Base tag : ${TAG}"
echo "Images   : ${images_to_build[*]}"
echo "Variants : ${variants_to_build[*]}"
echo ""

for image in "${images_to_build[@]}"; do
  # sanity check image
  if [ -z "${AMD64_DOCKERFILE[$image]+x}" ]; then
    echo "Unknown image: ${image} (skipped)"
    continue
  fi

  for variant in "${variants_to_build[@]}"; do
    arm64_df="${ARM64_VARIANT_DOCKERFILE[$variant]:-}"
    [ -n "${arm64_df}" ] || die "No arm64 Dockerfile configured for variant '${variant}'"

    echo "------------------------------------------------------------"
    echo "Building image='${image}' variant='${variant}'"
    echo "AMD64 Dockerfile: ${AMD64_DOCKERFILE[$image]}"
    echo "ARM64 Dockerfile: ${arm64_df}"
    echo "------------------------------------------------------------"

    build_multiarch_image_for_variant "${image}" "${variant}" "${arm64_df}"
  done
done

echo ""
echo "Done."
