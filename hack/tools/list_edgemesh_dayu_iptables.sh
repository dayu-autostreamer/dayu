#!/bin/bash

set -o errexit
set -o nounset
set -o pipefail

KEYWORD="dayu"
PROBE_TIMEOUT_SEC="${PROBE_TIMEOUT_SEC:-10}"

RAW_FILE=""
UNIQUE_FILE=""
SUMMARY_FILE=""

show_help() {
    cat <<'EOF'
Usage:
  ./hack/tools/list_edgemesh_dayu_iptables.sh [--keyword <substring>] [--timeout <seconds>]

Description:
  Inspect all Ready edgemesh-agent pods, collect nat table rules, and output
  only aggregated statistics for entries whose iptables comment namespace
  contains the given substring (default: "dayu").

Output columns:
  DEVICE        Kubernetes node that hosts the residual rule
  NAMESPACE     Namespace parsed from iptables comment
  ENTRY_COUNT   Number of unique residual entries on the device+namespace
  SERVICE_COUNT Number of unique residual services on the device+namespace
  SERVICES      Comma-separated residual service names
  PORTS         Comma-separated exposed ports
EOF
}

cleanup() {
    rm -f "${RAW_FILE:-}" "${UNIQUE_FILE:-}" "${SUMMARY_FILE:-}"
}

run_with_timeout() {
    local seconds="$1"
    shift

    if command -v timeout >/dev/null 2>&1; then
        timeout "${seconds}" "$@"
        return $?
    fi

    if command -v gtimeout >/dev/null 2>&1; then
        gtimeout "${seconds}" "$@"
        return $?
    fi

    "$@" &
    local pid=$!
    local start_ts
    start_ts="$(date +%s)"

    while kill -0 "${pid}" >/dev/null 2>&1; do
        if (( $(date +%s) - start_ts >= seconds )); then
            kill "${pid}" >/dev/null 2>&1 || true
            sleep 0.2
            kill -9 "${pid}" >/dev/null 2>&1 || true
            wait "${pid}" >/dev/null 2>&1 || true
            return 124
        fi
        sleep 0.2
    done

    wait "${pid}"
    return $?
}

require_cmd() {
    local cmd="$1"
    if ! command -v "${cmd}" >/dev/null 2>&1; then
        echo "Error: '${cmd}' is required but not found in PATH." >&2
        exit 1
    fi
}

list_edgemesh_agents() {
    kubectl get pods -A --no-headers \
        -o custom-columns=NS:.metadata.namespace,NAME:.metadata.name,NODE:.spec.nodeName,PHASE:.status.phase,READY:.status.containerStatuses[0].ready 2>/dev/null \
        | awk '$2 ~ /^edgemesh-agent/ && $4=="Running" && $5=="true" {print $1 "\t" $2 "\t" $3}'
}

extract_comment() {
    local rule="$1"
    printf '%s\n' "${rule}" \
        | sed -n \
            -e 's/.*--comment "\([^"]*\)".*/\1/p' \
            -e 's/.*--comment \([^[:space:]]*\).*/\1/p' \
        | head -n 1
}

extract_dport() {
    local rule="$1"
    printf '%s\n' "${rule}" \
        | sed -n \
            -e 's/.*--dport \([0-9][0-9]*\).*/\1/p' \
            -e 's/.*dpt:\([0-9][0-9]*\).*/\1/p' \
        | head -n 1
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --keyword|-k)
                KEYWORD="${2:-}"
                shift 2
                ;;
            --timeout|-t)
                PROBE_TIMEOUT_SEC="${2:-}"
                shift 2
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                echo "Unknown argument: $1" >&2
                show_help
                exit 1
                ;;
        esac
    done

    if [[ -z "${KEYWORD}" ]]; then
        echo "Error: --keyword cannot be empty." >&2
        exit 1
    fi

    if ! [[ "${PROBE_TIMEOUT_SEC}" =~ ^[0-9]+$ ]] || [[ "${PROBE_TIMEOUT_SEC}" -le 0 ]]; then
        echo "Error: --timeout must be a positive integer." >&2
        exit 1
    fi
}

main() {
    parse_args "$@"
    require_cmd kubectl

    RAW_FILE="$(mktemp)"
    UNIQUE_FILE="$(mktemp)"
    SUMMARY_FILE="$(mktemp)"
    trap cleanup EXIT

    local found=0
    local agent_count=0
    local probe_output=""

    while IFS=$'\t' read -r agent_ns agent_pod node_name; do
        [[ -z "${agent_ns}" ]] && continue
        agent_count=$((agent_count + 1))

        probe_output="$(
            run_with_timeout "${PROBE_TIMEOUT_SEC}" \
                kubectl --request-timeout="${PROBE_TIMEOUT_SEC}s" exec -n "${agent_ns}" "${agent_pod}" -- sh -c \
                '(iptables -t nat -S 2>/dev/null || iptables-save -t nat 2>/dev/null || true)' \
                2>/dev/null || true
        )"

        [[ -z "${probe_output}" ]] && continue

        while IFS= read -r rule; do
            local comment=""
            local dayu_ns=""
            local service_proto=""
            local service_name=""
            local node_port=""

            [[ "${rule}" == *"--comment"* ]] || continue

            comment="$(extract_comment "${rule}")"
            [[ -z "${comment}" ]] && continue
            [[ "${comment}" == */* ]] || continue

            dayu_ns="${comment%%/*}"
            [[ "${dayu_ns}" == *"${KEYWORD}"* ]] || continue

            service_proto="${comment#*/}"
            service_name="${service_proto%%:*}"
            node_port="$(extract_dport "${rule}")"
            [[ -n "${node_port}" ]] || node_port="-"

            printf '%s\t%s\t%s\t%s\n' \
                "${node_name}" \
                "${dayu_ns}" \
                "${service_name}" \
                "${node_port}" >> "${RAW_FILE}"
            found=$((found + 1))
        done <<< "${probe_output}"
    done < <(list_edgemesh_agents)

    if [[ "${agent_count}" -eq 0 ]]; then
        echo "No Ready edgemesh-agent pods found."
        exit 0
    fi

    if [[ "${found}" -eq 0 ]]; then
        echo "No iptables entries found whose namespace contains '${KEYWORD}'."
        exit 0
    fi

    sort -u "${RAW_FILE}" > "${UNIQUE_FILE}"

    local unique_count
    unique_count="$(wc -l < "${UNIQUE_FILE}" | tr -d ' ')"

    awk -F'\t' '
        {
            device = $1
            namespace = $2
            service = $3
            port = $4
            group = device SUBSEP namespace

            if (!(group in group_seen)) {
                group_seen[group] = 1
                order[++order_count] = group
            }

            entry_count[group]++

            service_key = group SUBSEP service
            if (!(service_key in seen_service)) {
                seen_service[service_key] = 1
                service_count[group]++
                services[group] = services[group] ? services[group] "," service : service
            }

            port_key = group SUBSEP port
            if (!(port_key in seen_port)) {
                seen_port[port_key] = 1
                ports[group] = ports[group] ? ports[group] "," port : port
            }
        }
        END {
            print "DEVICE\tNAMESPACE\tENTRY_COUNT\tSERVICE_COUNT\tSERVICES\tPORTS"
            for (i = 1; i <= order_count; i++) {
                group = order[i]
                split(group, parts, SUBSEP)
                print parts[1] "\t" parts[2] "\t" entry_count[group] "\t" service_count[group] "\t" services[group] "\t" ports[group]
            }
        }
    ' "${UNIQUE_FILE}" > "${SUMMARY_FILE}"

    echo "Found ${unique_count} unique matching residual entries on ${agent_count} Ready edgemesh-agent pod(s)."
    echo

    if command -v column >/dev/null 2>&1; then
        column -t -s $'\t' "${SUMMARY_FILE}"
    else
        cat "${SUMMARY_FILE}"
    fi
}

main "$@"
