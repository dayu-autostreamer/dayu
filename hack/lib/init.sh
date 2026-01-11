#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

# The root of the dayu
DAYU_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"

source "${DAYU_ROOT}/hack/lib/utils.sh"
source "${DAYU_ROOT}/hack/lib/buildx.sh"
