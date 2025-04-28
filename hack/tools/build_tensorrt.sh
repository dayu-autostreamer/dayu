#!/bin/bash

# Function: Display help information
show_help() {
cat << EOF
Usage: ${0##*/} [--files [tensorrt...]] [--tag TAG] [--repo REPO] [--no-cache] [--help]

--files        Specify the images to build, separated by commas. Options include:
               tensorrt
               Default is to select all.
--tag          Specify the version tag for the Docker images. Default is "trt8".
--repo         Specify the repository for the Docker images. Default is "dayuhub".
--registry     Specify the registry of docker.
--no-cache     Build the Docker images without using cache.
--help         Display this help message and exit.
EOF
}

# Default Dockerfiles and their platforms
declare -A DOCKERFILES=(
    [tensorrt]="build/tensorrt_[amd64/arm64].Dockerfile"
)

# Corresponding platforms
declare -A PLATFORMS=(
    [tensorrt]="linux/amd64,linux/arm64"
)

# Images requiring special treatment, their platforms, and Dockerfiles
declare -A SPECIAL_BUILD=(
    [tensorrt]="linux/amd64:build/tensorrt_amd64.Dockerfile,linux/arm64:build/tensorrt_arm64.Dockerfile"

)

# Initialize variables
SELECTED_FILES=""
TAG="trt8"  # Default tag
REPO="dayuhub"  # Default repository
NO_CACHE=false  # Default is to use cache
REGISTRY="${REG:-docker.io}"

# Parse command line arguments
while :; do
    case $1 in
        --help)
            show_help    # Display help information
            exit 0
            ;;
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

build_image() {
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

build_image_special() {
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

create_and_push_manifest() {
    local image=$1
    local tag=$2
    local repo=$3
    local manifest_tag="${REGISTRY}/${repo}/${image}:${tag}"

    echo "Creating and pushing manifest for: $manifest_tag"

    docker buildx imagetools create -t "$manifest_tag" \
        "${REGISTRY}/${repo}/${image}:${tag}-amd64" \
        "${REGISTRY}/${repo}/${image}:${tag}-arm64"

}

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
            if [[ -n "${SPECIAL_BUILD[$image]}" ]]; then
                IFS=',' read -ra SPECIAL_PLATFORMS <<< "${SPECIAL_BUILD[$image]}"
                for entry in "${SPECIAL_PLATFORMS[@]}"; do
                    IFS=':' read -ra DETAILS <<< "$entry"
                    platform="${DETAILS[0]}"
                    dockerfile="${DETAILS[1]}"
                    build_image_special "$image" "$platform" "$dockerfile" "$CACHE_OPTION"
                done
                # After building all architectures, create and push manifest
                create_and_push_manifest "$image" "$TAG" "$REPO"
            else
                build_image "$image" "${PLATFORMS[$image]}" "${DOCKERFILES[$image]}" "$CACHE_OPTION"
            fi
        else
            echo "Unknown image or platform not specified: $image"
        fi
    done
else
    echo "No images specified, building all default images."
    for image in "${!DOCKERFILES[@]}"; do
        if [[ -n "${SPECIAL_BUILD[$image]}" ]]; then
            IFS=',' read -ra SPECIAL_PLATFORMS <<< "${SPECIAL_BUILD[$image]}"
            for entry in "${SPECIAL_PLATFORMS[@]}"; do
                IFS=':' read -ra DETAILS <<< "$entry"
                platform="${DETAILS[0]}"
                dockerfile="${DETAILS[1]}"
                build_image_special "$image" "$platform" "$dockerfile" "$CACHE_OPTION"
            done
            # After building all architectures, create and push manifest
            create_and_push_manifest "$image" "$TAG" "$REPO"
        else
            build_image "$image" "${PLATFORMS[$image]}" "${DOCKERFILES[$image]}" "$CACHE_OPTION"
        fi
    done
fi

