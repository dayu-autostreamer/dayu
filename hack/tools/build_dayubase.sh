#!/usr/bin/env bash
set -o errexit
set -o nounset
set -o pipefail

DAYU_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
source "${DAYU_ROOT}/hack/lib/init.sh"

show_help() {
  cat << EOF
Usage: ${0##*/} [--files dayubase] [--jp JP] [--tag TAG] [--repo REPO] [--registry REG]
                 [--no-cache] [--print] [--help]

--files        Specify images to build. Only "dayubase" is supported.
--jp           Select JetPack variant(s): default, jp4/4, jp5/5, jp6/6, all.
               Default: default. Comma-separated values are accepted.
--tag          Base tag. Default: latest. JP tags become TAG-jpX.
--repo         Output repository/namespace. Default: dayuhub.
--registry     Output registry. Default: \${REG:-docker.io}.
--no-cache     Disable Docker build cache.
--print        Print the resolved Bake targets instead of building.
--help         Display this help message and exit.

This script builds the dayubase arch-specific targets from docker-bake.hcl and
then creates the final multi-arch manifest tags.
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

normalize_variant() {
  local variant="$1"
  variant="$(echo "$variant" | tr '[:upper:]' '[:lower:]')"
  case "$variant" in
    default) echo "default" ;;
    all) echo "all" ;;
    jp4|4) echo "jp4" ;;
    jp5|5) echo "jp5" ;;
    jp6|6) echo "jp6" ;;
    *) die "Unknown --jp variant: '$1' (allowed: default, jp4/jp5/jp6, 4/5/6, all)" ;;
  esac
}

final_tag_for_variant() {
  local variant="$1"
  if [[ "$variant" == "default" ]]; then
    echo "$TAG"
  else
    echo "${TAG}-${variant}"
  fi
}

SELECTED_FILES=""
JP_VARIANTS_RAW=""
TAG="${TAG:-latest}"
REPO="${REPO:-dayuhub}"
REGISTRY="${REGISTRY:-${REG:-docker.io}}"
BASE_REPO="${BASE_REPO:-dayuhub}"
BASE_TAG="${BASE_TAG:-latest}"
NO_CACHE=false
PRINT_ONLY=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help)
      show_help
      exit 0
      ;;
    --files)
      [[ -n "${2:-}" ]] || die '"--files" requires a non-empty option argument.'
      SELECTED_FILES="$2"
      shift
      ;;
    --jp)
      [[ -n "${2:-}" ]] || die '"--jp" requires a non-empty option argument.'
      JP_VARIANTS_RAW="$2"
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

if [[ -n "$SELECTED_FILES" && "$SELECTED_FILES" != "dayubase" ]]; then
  die "Only --files dayubase is supported by ${0##*/}."
fi

variants_to_build=()
if [[ -z "$JP_VARIANTS_RAW" ]]; then
  variants_to_build=("default")
else
  IFS=',' read -ra raw_variants <<< "$JP_VARIANTS_RAW"
  expanded=false
  for token in "${raw_variants[@]}"; do
    normalized="$(normalize_variant "$token")"
    if [[ "$normalized" == "all" ]]; then
      variants_to_build=("default" "jp4" "jp5" "jp6")
      expanded=true
      break
    fi
  done

  if [[ "$expanded" == "false" ]]; then
    declare -A seen=()
    for token in "${raw_variants[@]}"; do
      normalized="$(normalize_variant "$token")"
      [[ "$normalized" == "all" ]] && continue
      if [[ -z "${seen[$normalized]+x}" ]]; then
        variants_to_build+=("$normalized")
        seen[$normalized]=1
      fi
    done
  fi
fi

echo "Registry : ${REGISTRY}"
echo "Repo     : ${REPO}"
echo "Base tag : ${TAG}"
echo "Variants : ${variants_to_build[*]}"
echo "No-cache : ${NO_CACHE}"
echo ""

if [[ "$PRINT_ONLY" != "true" ]]; then
  dayu::buildx::prepare_env
fi

export REGISTRY REPO TAG BASE_REPO BASE_TAG

bake_options=(-f "${DAYU_ROOT}/docker-bake.hcl")
if [[ "$NO_CACHE" == "true" ]]; then
  bake_options+=(--set=*.no-cache=true)
fi
if [[ "$PRINT_ONLY" == "true" ]]; then
  bake_options+=(--print)
fi

for variant in "${variants_to_build[@]}"; do
  final_tag="$(final_tag_for_variant "$variant")"
  amd64_target="dayubase-${variant}-amd64"
  arm64_target="dayubase-${variant}-arm64"

  echo "------------------------------------------------------------"
  echo "Building dayubase variant='${variant}' final tag='${final_tag}'"
  echo "Bake targets: ${amd64_target} ${arm64_target}"
  echo "------------------------------------------------------------"

  docker buildx bake "${bake_options[@]}" "$amd64_target" "$arm64_target"

  if [[ "$PRINT_ONLY" != "true" ]]; then
    manifest_tag="${REGISTRY}/${REPO}/dayubase:${final_tag}"
    echo "Creating and pushing manifest: ${manifest_tag}"
    docker buildx imagetools create -t "$manifest_tag" \
      "${REGISTRY}/${REPO}/dayubase:${final_tag}-amd64" \
      "${REGISTRY}/${REPO}/dayubase:${final_tag}-arm64"
  fi
done

echo ""
echo "Done."
