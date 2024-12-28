#!/usr/bin/env bash


set -o errexit
set -o nounset
set -o pipefail

DAYU_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
source "${DAYU_ROOT}/hack/lib/init.sh"

dayu::buildx::build_and_push_multi_platform_images "$@"