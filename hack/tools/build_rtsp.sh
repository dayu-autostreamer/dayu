#!/usr/bin/env bash
set -o errexit
set -o nounset
set -o pipefail

DAYU_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"

has_files=false
for arg in "$@"; do
  if [[ "$arg" == "--files" ]]; then
    has_files=true
    break
  fi
done

if [[ "$has_files" == "true" ]]; then
  exec bash "${DAYU_ROOT}/hack/make-rules/cross-build.sh" "$@"
else
  exec bash "${DAYU_ROOT}/hack/make-rules/cross-build.sh" --files rtsp-server "$@"
fi
