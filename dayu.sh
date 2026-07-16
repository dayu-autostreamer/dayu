#!/bin/bash

set -o errexit
set -o nounset
set -o pipefail

NO_COLOR='\033[0m'
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'

green_text() {
  echo -ne "$GREEN$@$NO_COLOR"
}
yellow_text() {
  echo -ne "$YELLOW$@$NO_COLOR"
}
red_text() {
  echo -ne "$RED$@$NO_COLOR"
}

check_install_yq() {
    if ! command -v yq &> /dev/null; then
        echo "yq could not be found, installing..."
        wget -O /usr/local/bin/yq https://github.com/mikefarah/yq/releases/download/v4.6.1/yq_linux_amd64
        chmod +x /usr/local/bin/yq
        echo "yq installed successfully."
    fi
}

check_namespace_existence() {
    kubectl get namespace "$NAMESPACE" > /dev/null 2>&1
}

check_and_create_namespace() {
    if check_namespace_existence; then

        echo "Namespace $(red_text "$NAMESPACE") already exists. Please use ACTION=stop to clean up before start system."
        exit 1
    else
        echo "$(green_text [DAYU]) Creating namespace $NAMESPACE..."
        kubectl create namespace "$NAMESPACE"
    fi
}

create_service_account() {
    echo "$(green_text [DAYU]) Creating the backend-only Kubernetes service account..."
    kubectl apply -f - <<EOF
apiVersion: v1
kind: ServiceAccount
metadata:
  name: $BACKEND_SERVICE_ACCOUNT
  namespace: $NAMESPACE
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: $NAMESPACE
  name: $BACKEND_ROLE
rules:
  - apiGroups: ["sedna.io"]
    resources: ["runtimeservices"]
    verbs: ["get", "list", "watch", "create", "delete"]
  - apiGroups: [""]
    resources: ["configmaps"]
    verbs: ["get", "create", "update", "delete"]
  - apiGroups: ["metrics.k8s.io"]
    resources: ["pods"]
    verbs: ["list"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  namespace: $NAMESPACE
  name: $BACKEND_ROLE_BINDING
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: $BACKEND_ROLE
subjects:
  - kind: ServiceAccount
    name: $BACKEND_SERVICE_ACCOUNT
    namespace: $NAMESPACE
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: $BACKEND_CLUSTER_ROLE
rules:
  - apiGroups: [""]
    resources: ["nodes"]
    verbs: ["list"]
  - apiGroups: [""]
    resources: ["pods"]
    verbs: ["list"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: $BACKEND_CLUSTER_ROLE_BINDING
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: $BACKEND_CLUSTER_ROLE
subjects:
  - kind: ServiceAccount
    name: $BACKEND_SERVICE_ACCOUNT
    namespace: $NAMESPACE
EOF
}

create_redis() {
  echo "$(green_text [DAYU]) Creating redis ..."
      kubectl -n "$NAMESPACE" apply -f - <<EOF
apiVersion: $SUPPORT_API_VERSION
kind: $SUPPORT_KIND
metadata:
  name: redis
  namespace: $NAMESPACE
spec:
  cloudWorker:
    mounts:
      - name: redis-runtime-state
        source:
          type: hostPath
          hostPath:
            path: runtime-state/$NAMESPACE/redis
            pathType: DirectoryOrCreate
            prefix: $DEFAULT_FILE_MOUNT_PREFIX
        target:
          path: /data
    logLevel:
      level: "DEBUG"
    template:
      spec:
        containers:
          - image: $REGISTRY/redis:latest
            imagePullPolicy: Always
            name: redis
            args: ["--appendonly", "yes", "--appendfsync", "always", "--dir", "/data"]
            ports:
              - containerPort: 6379
        dnsPolicy: ClusterFirstWithHostNet
        enableServiceLinks: false
        nodeName: $CLOUD_NODE
        automountServiceAccountToken: false
  serviceConfig:
    port: 6379
    pos: cloud
    targetPort: 6379
EOF
}

create_datasource() {
  if [ "$DATASOURCE_USE_SIMULATION" = "true" ]; then
    echo "$(green_text [DAYU]) Creating datasource ..."
    kubectl -n "$NAMESPACE" apply -f - <<EOF
apiVersion: $SUPPORT_API_VERSION
kind: $SUPPORT_KIND
metadata:
  name: datasource
  namespace: $NAMESPACE
spec:
  edgeWorker:
    - mounts:
        - source:
            type: hostPath
            hostPath:
              path: $DATASOURCE_DATA_ROOT
              pathType: Directory
          envName: DEFAULT_MOUNT_PATH
        - name: temporary-directory
          source:
            type: hostPath
            hostPath:
              path: temp/
              pathType: DirectoryOrCreate
              prefix: $DEFAULT_FILE_MOUNT_PREFIX
          target:
            path: /temp
          envName: TEMP_PATH
      logLevel:
        level: "DEBUG"
      template:
        spec:
          containers:
            - env:
                - name: REQUEST_INTERVAL
                  value: "2"
                - name: START_INTERVAL
                  value: "4"
                - name: PLAY_MODE
                  value: "$DATASOURCE_PLAY_MODE"
                - name: DAYU_BACKEND_ENDPOINT
                  value: "http://backend-cloud.$NAMESPACE.svc.cluster.local.:8000"
                - name: GUNICORN_PORT
                  value: "8000"
              image: $REGISTRY/$REPOSITORY/datasource:$TAG
              imagePullPolicy: Always
              name: datasource
              ports:
                - containerPort: 8000
          dnsPolicy: ClusterFirstWithHostNet
          enableServiceLinks: false
          nodeName: $DATASOURCE_NODE
          automountServiceAccountToken: false
  serviceConfig:
    port: 8000
    pos: edge
    targetPort: 8000
EOF

    else
        echo "Skipping creation of datasource since DATASOURCE_USE_SIMULATION is false."
    fi
}

create_backend() {
  echo "$(green_text [DAYU]) Creating backend ..."
      kubectl -n "$NAMESPACE" apply -f - <<EOF
apiVersion: $SUPPORT_API_VERSION
kind: $SUPPORT_KIND
metadata:
  name: backend
  namespace: $NAMESPACE
spec:
  cloudWorker:
    mounts:
      - source:
          type: hostPath
          hostPath:
            path: $TEMPLATE
            pathType: Directory
        envName: DEFAULT_MOUNT_PATH
      - name: temporary-directory
        source:
          type: hostPath
          hostPath:
            path: temp/
            pathType: DirectoryOrCreate
            prefix: $DEFAULT_FILE_MOUNT_PREFIX
        target:
          path: /temp
        envName: TEMP_PATH
    logLevel:
      level: "DEBUG"
    template:
      spec:
        containers:
          - env:
            - name: GUNICORN_PORT
              value: "8000"
            - name: CLOUD_NODE_NAME
              value: "$CLOUD_NODE"
            - name: DAYU_RUNTIME_CONTROL_PLANE
              value: "true"
            - name: SYSTEM_LOG_RETENTION_RECORDS
              value: "$SYSTEM_LOG_RETENTION_RECORDS"
            - name: SYSTEM_LOG_COMPACT_INTERVAL
              value: "$SYSTEM_LOG_COMPACT_INTERVAL"
            image: $REGISTRY/$REPOSITORY/backend:$TAG
            imagePullPolicy: Always
            name: backend
            ports:
              - containerPort: 8000
        dnsPolicy: ClusterFirstWithHostNet
        enableServiceLinks: false
        nodeName: $CLOUD_NODE
        serviceAccountName: $BACKEND_SERVICE_ACCOUNT
  serviceConfig:
    port: 8000
    pos: cloud
    targetPort: 8000
EOF

}

create_frontend() {
  echo "$(green_text [DAYU]) Creating frontend ..."
      kubectl -n "$NAMESPACE" apply -f - <<EOF
apiVersion: $SUPPORT_API_VERSION
kind: $SUPPORT_KIND
metadata:
  name: frontend
  namespace: $NAMESPACE
spec:
  cloudWorker:
    logLevel:
      level: "DEBUG"
    template:
      spec:
        containers:
          - env:
            - name: VITE_DAYU_VERSION
              value: $TAG
            - name: VITE_BACKEND_ADDRESS
              value: 'http://backend-cloud.$NAMESPACE.svc.cluster.local.:8000'
            - name: VITE_PORT
              value: '8000'
            - name: VITE_OPEN
              value: 'false'
            - name: VITE_OPEN_CDN
              value: 'false'
            - name: VITE_PUBLIC_PATH
              value: /vue-next-admin-preview/
            image: $REGISTRY/$REPOSITORY/frontend:$TAG
            imagePullPolicy: Always
            name: frontend
            ports:
              - containerPort: 8000
        dnsPolicy: ClusterFirstWithHostNet
        enableServiceLinks: false
        nodeName: $CLOUD_NODE
        automountServiceAccountToken: false
  serviceConfig:
    port: 8000
    pos: cloud
    targetPort: 8000
EOF

}

wait_for_pods_running() {
    local namespace=$NAMESPACE
    local timeout=120
    local start_time=$(date +%s)

    echo "$(green_text [DAYU]) Waiting for all pods in namespace '$namespace' to be in the 'Running' state..."

    while true; do
        local non_running_pods=$(kubectl get pods -n "$namespace" --no-headers | grep -v "Running" | wc -l)
        if [[ "$non_running_pods" -eq 0 ]]; then
            echo "All pods are in the 'Running' state in namespace '$namespace'."
            return
        else
            local current_time=$(date +%s)
            local elapsed_time=$((current_time - start_time))
            if [[ "$elapsed_time" -ge "$timeout" ]]; then
                echo "Pods initialize $(red_text timeout). Use 'kubectl get pods -n $namespace' to see details."
                exit 1
            fi

            sleep 2
        fi
    done
}

start_system() {
    echo "$(green_text [DAYU]) Starting DAYU system in namespace $NAMESPACE..."
    check_and_create_namespace
    create_service_account
    create_redis
    create_backend
    create_frontend
    create_datasource
    wait_for_pods_running
    show_prompt_infos
}

delete_service_account() {
  _kubectl_delete clusterrolebinding "$BACKEND_CLUSTER_ROLE_BINDING" --ignore-not-found=true
  _kubectl_delete clusterrole "$BACKEND_CLUSTER_ROLE" --ignore-not-found=true
  _kubectl_delete rolebinding "$BACKEND_ROLE_BINDING" -n "$NAMESPACE" --ignore-not-found=true
  _kubectl_delete role "$BACKEND_ROLE" -n "$NAMESPACE" --ignore-not-found=true
  _kubectl_delete serviceaccount "$BACKEND_SERVICE_ACCOUNT" -n "$NAMESPACE" --ignore-not-found=true
}

stop_system() {
    local ns="${NAMESPACE}"
    local mesh_wait="${MESH_WAIT_SEC:-30}"
    local ns_wait="${NS_WAIT_SEC:-120}"
    # The backend command is asynchronous. Give its fast exact-UID teardown a
    # short bounded window, then preserve the historical stop contract by
    # continuing with shell cleanup instead of waiting on a broken control path.
    local graceful_wait="${GRACEFUL_STOP_WAIT_SEC:-60}"
    local wait_mesh_rules="${WAIT_EDGEMESH_RULES:-true}"
    local app_resources=""

    echo "$(green_text [DAYU]) Stopping DAYU system in namespace ${ns}..."

    # ---------------- helper: run a command with timeout (best-effort, portable) ----------------
    _run_with_timeout() {
        local seconds="$1"; shift

        # Prefer GNU coreutils timeout if available.
        if command -v timeout >/dev/null 2>&1; then
            timeout "${seconds}" "$@"
            return $?
        fi
        # macOS users may have gtimeout via coreutils.
        if command -v gtimeout >/dev/null 2>&1; then
            gtimeout "${seconds}" "$@"
            return $?
        fi

        # Fallback: run in background and kill if it exceeds the budget.
        "$@" &
        local pid=$!
        local start_ts
        start_ts="$(date +%s)"

        while kill -0 "${pid}" >/dev/null 2>&1; do
            if (( $(date +%s) - start_ts >= seconds )); then
                kill "${pid}" >/dev/null 2>&1 || true
                # Try harder if it doesn't stop quickly.
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

    _kubectl_read() {
        _run_with_timeout 6 kubectl --request-timeout=5s "$@"
    }

    _kubectl_delete() {
        _run_with_timeout 11 kubectl --request-timeout=10s delete "$@" --wait=false
    }

    local namespace_state=""
    if ! namespace_state="$(_kubectl_read get namespace "${ns}" \
            --ignore-not-found=true -o name 2>/dev/null)"; then
        echo "$(yellow_text [DAYU]) Unable to verify namespace '${ns}'; system stop cannot continue safely."
        return 1
    fi
    if [[ -z "${namespace_state}" ]]; then
        echo "$(green_text [DAYU]) Namespace '${ns}' is already absent; remove deployment-scoped access bindings."
        delete_service_account || true
        echo "$(green_text [DAYU]) DAYU system is already stopped."
        return 0
    fi

    _bool_is_true() {
        case "${1:-}" in
            1|true|TRUE|True|yes|YES|Yes|on|ON|On)
                return 0
                ;;
            *)
                return 1
                ;;
        esac
    }

    _list_dayu_app_resources() {
        local namespace="$1"
        _kubectl_read get runtimeservices.sedna.io -n "${namespace}" \
            -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' 2>/dev/null \
            || true
    }

    # ---------------- helper: list Ready edgemesh-agent pods (ns/pod) ----------------
    _list_edgemesh_pods() {
        # Only consider Running/Ready pods to avoid kubectl exec hanging on terminating/unready agents.
        _kubectl_read get pods -A --no-headers \
            -o custom-columns=NS:.metadata.namespace,NAME:.metadata.name,PHASE:.status.phase,READY:.status.containerStatuses[0].ready 2>/dev/null \
          | awk '$2 ~ /^edgemesh-agent/ && $3=="Running" && $4=="true" {print $1"/"$2}'
    }

    # ---------------- helper: check if any edgemesh-agent still has iptables rules for this namespace ----------------
    _edgemesh_has_rules() {
        local namespace="$1"
        local pods
        pods="$(_list_edgemesh_pods || true)"

        # No edgemesh-agent, as no rules
        [[ -z "${pods}" ]] && return 1

        while read -r item; do
            [[ -z "${item}" ]] && continue
            local pns="${item%%/*}"
            local pname="${item##*/}"

            # NOTE:
            # - "kubectl exec" itself can hang if apiserver->kubelet tunnel is broken.
            # - Bound the probe, so outer mesh_wait timeout will actually work.
            # - Extract the namespace part from comment "namespace/service" and
            #   compare it exactly, to avoid confusion like "dayu" vs "dayu-xxx".
            # - Treat probe failures as NOT blocking stop (return 1 for this pod).
            if _run_with_timeout 5 \
                kubectl --request-timeout=5s exec -n "${pns}" "${pname}" -- sh -c \
                "iptables -t nat -S 2>/dev/null \
                | sed -n \
                    -e 's/.*--comment \"\\([^\"]*\\)\".*/\\1/p' \
                    -e 's/.*--comment \\([^[:space:]]*\\).*/\\1/p' \
                | awk -F/ '\$1 == \"${namespace}\" { found=1; exit 0 } END { exit(found ? 0 : 1) }'" \
                >/dev/null 2>&1; then
                return 0
            fi

        done <<< "${pods}"

        return 1
    }

    # ---------------- helper: wait until edgemesh removes iptables rules for this namespace ----------------
    _wait_edgemesh_rules_gone() {
        local namespace="$1"
        local timeout="$2"
        local pods
        pods="$(_list_edgemesh_pods || true)"
        if [[ -z "${pods}" ]]; then
            echo "$(green_text [DAYU]) No Ready edgemesh-agent pods found, skip EdgeMesh rule wait."
            return 0
        fi

        local start_ts
        start_ts="$(date +%s)"

        while true; do
            if ! _edgemesh_has_rules "${namespace}"; then
                echo "$(green_text [DAYU]) OK: EdgeMesh iptables rules for '${namespace}' are gone."
                return 0
            fi

            if (( $(date +%s) - start_ts > timeout )); then
                echo "EdgeMesh iptables rules for '${namespace}' still exist after ${timeout}s"
                return 1
            fi
            sleep 2
        done
    }

    _try_backend_stop_service() {
        local namespace="$1"
        local backend_service="backend-cloud"
        local backend_port
        local backend_url
        local backend_state_url
        local response=""
        local state_response=""
        local parsed_state=""
        local parsed_phase=""
        local parsed_snapshot=""
        local install_id=""
        local current_install_id=""
        local stop_payload=""
        local http_client=""
        local start_ts

        if ! _kubectl_read get svc "${backend_service}" -n "${namespace}" >/dev/null 2>&1; then
            echo "$(yellow_text [DAYU]) Backend service '${backend_service}' not found, skip graceful service uninstall."
            return 1
        fi

        backend_port="$(_kubectl_read get svc "${backend_service}" -n "${namespace}" \
            -o=jsonpath='{.spec.ports[0].nodePort}' 2>/dev/null || true)"
        if [[ -z "${backend_port}" ]]; then
            echo "$(yellow_text [DAYU]) Failed to resolve backend NodePort, skip graceful service uninstall."
            return 1
        fi

        backend_url="http://${CLOUD_IP}:${backend_port}/stop_service"
        backend_state_url="http://${CLOUD_IP}:${backend_port}/install_state"
        echo "$(green_text [DAYU]) Try graceful service uninstall via backend API: ${backend_url}"
        start_ts="$(date +%s)"

        if command -v curl >/dev/null 2>&1; then
            http_client="curl"
            state_response="$(_run_with_timeout 5 \
                curl --silent --show-error --max-time 5 "${backend_state_url}" 2>/dev/null || true)"
        elif command -v wget >/dev/null 2>&1; then
            http_client="wget"
            state_response="$(_run_with_timeout 5 \
                wget -qO- --timeout=5 "${backend_state_url}" 2>/dev/null || true)"
        else
            echo "$(yellow_text [DAYU]) Neither curl nor wget found, skip graceful service uninstall."
            return 1
        fi

        if parsed_snapshot="$(_parse_install_state "${state_response}")"; then
            IFS=$'\t' read -r parsed_state parsed_phase install_id <<< "${parsed_snapshot}"
            if [[ "${parsed_state}" == "uninstall" \
                    && "${parsed_phase}" == "uninstalled" \
                    && -z "${install_id}" ]]; then
                echo "$(green_text [DAYU]) No managed runtime or install admission is active."
                return 0
            fi
        else
            echo "$(yellow_text [DAYU]) Backend install state is unavailable or invalid; use the trusted global stop fallback."
        fi

        if [[ -n "${install_id}" ]]; then
            stop_payload="{\"install_id\":\"${install_id}\"}"
        fi

        if [[ "${http_client}" == "curl" ]]; then
            if [[ -n "${stop_payload}" ]]; then
                response="$(_run_with_timeout "${graceful_wait}" \
                    curl --silent --show-error --max-time "${graceful_wait}" -X POST \
                    -H 'Content-Type: application/json' --data "${stop_payload}" \
                    "${backend_url}" 2>/dev/null || true)"
            else
                response="$(_run_with_timeout "${graceful_wait}" \
                    curl --silent --show-error --max-time "${graceful_wait}" -X POST \
                    "${backend_url}" 2>/dev/null || true)"
            fi
        elif [[ -n "${stop_payload}" ]]; then
            response="$(_run_with_timeout "${graceful_wait}" \
                wget -qO- --timeout="${graceful_wait}" --method=POST \
                --header='Content-Type: application/json' --body-data="${stop_payload}" \
                "${backend_url}" 2>/dev/null || true)"
        else
            response="$(_run_with_timeout "${graceful_wait}" \
                wget -qO- --timeout="${graceful_wait}" --method=POST \
                "${backend_url}" 2>/dev/null || true)"
        fi

        if [[ -z "${response}" ]]; then
            echo "$(yellow_text [DAYU]) Backend graceful uninstall returned no response, continue with shell cleanup."
            return 1
        fi

        if printf '%s' "${response}" | grep -q '"state"[[:space:]]*:[[:space:]]*"success"'; then
            while (( $(date +%s) - start_ts < graceful_wait )); do
                if [[ "${http_client}" == "curl" ]]; then
                    state_response="$(_run_with_timeout 5 \
                        curl --silent --show-error --max-time 5 "${backend_state_url}" 2>/dev/null || true)"
                else
                    state_response="$(_run_with_timeout 5 \
                        wget -qO- --timeout=5 "${backend_state_url}" 2>/dev/null || true)"
                fi
                if parsed_snapshot="$(_parse_install_state "${state_response}")"; then
                    IFS=$'\t' read -r parsed_state parsed_phase current_install_id <<< "${parsed_snapshot}"
                    if [[ -n "${install_id}" ]]; then
                        if [[ "${current_install_id}" != "${install_id}" ]]; then
                            echo "$(green_text [DAYU]) Backend graceful uninstall finished successfully."
                            return 0
                        fi
                    elif [[ "${parsed_state}" == "uninstall" \
                            && "${parsed_phase}" == "uninstalled" \
                            && -z "${current_install_id}" ]]; then
                        echo "$(green_text [DAYU]) Backend graceful uninstall finished successfully."
                        return 0
                    fi
                fi
                sleep 2
            done
            echo "$(yellow_text [DAYU]) Backend graceful uninstall exceeded ${graceful_wait}s; continue with shell cleanup."
            return 1
        fi

        echo "$(yellow_text [DAYU]) Backend graceful uninstall did not finish cleanly: ${response}"
        return 1
    }

    _parse_install_state() {
        local response="$1"
        local state
        local phase
        local install_id

        [[ -n "${response}" ]] || return 1
        state="$(printf '%s' "${response}" | yq e '.state' - 2>/dev/null)" || return 1
        phase="$(printf '%s' "${response}" | yq e '.phase' - 2>/dev/null)" || return 1
        install_id="$(printf '%s' "${response}" | yq e '.install_id' - 2>/dev/null)" || return 1

        [[ "${state}" == "install" || "${state}" == "uninstall" ]] || return 1
        [[ -n "${phase}" && "${phase}" != "null" ]] || return 1
        [[ "${install_id}" != "null" ]] || return 1

        if [[ -z "${install_id}" ]]; then
            [[ "${state}" == "uninstall" && "${phase}" == "uninstalled" ]] || return 1
        elif [[ "${phase}" == "uninstalled" ]]; then
            return 1
        elif ! [[ "${install_id}" =~ ^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$ ]]; then
            return 1
        fi

        printf '%s\t%s\t%s\n' "${state}" "${phase}" "${install_id}"
    }

    app_resources="$(_list_dayu_app_resources "${ns}")"

    echo "$(green_text [DAYU]) (0/6) Try graceful uninstall for deployed services..."
    if ! _try_backend_stop_service "${ns}"; then
        echo "$(yellow_text [DAYU]) Backend graceful uninstall failed; continue with system cleanup."
    fi

    echo "$(green_text [DAYU]) (1/6) Delete RuntimeServices, then bootstrap resources..."
    _kubectl_delete runtimeservices.sedna.io -n "${ns}" --all --ignore-not-found=true 2>/dev/null || true
    _kubectl_delete "${SUPPORT_KIND}" -n "${ns}" --all --ignore-not-found=true 2>/dev/null || true

    echo "$(green_text [DAYU]) (2/6) Delete Services/Endpoints..."
    _kubectl_delete svc -n "${ns}" --all --ignore-not-found=true 2>/dev/null || true
    _kubectl_delete endpoints -n "${ns}" --all --ignore-not-found=true 2>/dev/null || true
    _kubectl_delete endpointslices -n "${ns}" --all --ignore-not-found=true 2>/dev/null || true
    _kubectl_delete ingress -n "${ns}" --all --ignore-not-found=true 2>/dev/null || true

    echo "$(green_text [DAYU]) (3/6) Delete workloads and remaining resources in namespace '${ns}'..."
    _kubectl_delete deploy,sts,ds,rs,po,job,cronjob,hpa -n "${ns}" --all --ignore-not-found=true 2>/dev/null || true
    _kubectl_delete cm,secret -n "${ns}" --all --ignore-not-found=true 2>/dev/null || true

    if [[ -n "${app_resources}" ]] && _bool_is_true "${wait_mesh_rules}"; then
        echo "$(green_text [DAYU]) (4/6) Waiting EdgeMesh to remove iptables rules for '${ns}'..."
        if ! _wait_edgemesh_rules_gone "${ns}" "${mesh_wait}"; then
            echo "[WARN] EdgeMesh didn't cleanup rules in time, continue anyway."
        fi
    else
        echo "$(green_text [DAYU]) (4/6) Skip EdgeMesh rule wait (no DAYU services found or WAIT_EDGEMESH_RULES is disabled)."
    fi

    echo "$(green_text [DAYU]) (5/6) Delete service account binding..."
    delete_service_account || true
    _kubectl_delete role,rolebinding -n "${ns}" --all --ignore-not-found=true 2>/dev/null || true

    echo "$(green_text [DAYU]) (6/6) Delete namespace '${ns}'..."
    if ! _run_with_timeout "$((ns_wait + 10))" \
            kubectl --request-timeout=10s delete namespace "${ns}" \
            --ignore-not-found=true --wait=true --timeout="${ns_wait}s" 2>/dev/null; then
        echo "$(yellow_text [DAYU]) Namespace deletion command did not complete cleanly; verify the final namespace state."
    fi

    namespace_state=""
    if ! namespace_state="$(_kubectl_read get namespace "${ns}" \
            --ignore-not-found=true -o name 2>/dev/null)"; then
        echo "$(red_text [DAYU]) Unable to verify that namespace '${ns}' was removed."
        return 1
    fi
    if [[ -n "${namespace_state}" ]]; then
        echo "$(red_text [DAYU]) Namespace '${ns}' still exists; DAYU system stop is incomplete."
        return 1
    fi

    echo "$(green_text DAYU system stop successfully.)"
}

show_prompt_infos() {
  sleep 1
  FRONTEND_PORT=$(get_service_nodeport "frontend-cloud" "$NAMESPACE")
  echo "$(green_text "██████╗  █████╗ ██╗   ██╗██╗   ██╗")"
  echo "$(green_text "██╔══██╗██╔══██╗╚██╗ ██╔╝██║   ██║")"
  echo "$(green_text "██║  ██║███████║ ╚████╔╝ ██║   ██║")"
  echo "$(green_text "██║  ██║██╔══██║  ╚██╔╝  ╚██╗ ██╔╝")"
  echo "$(green_text "██████╔╝██║  ██║   ██║    ╚████╔╝ ")"
  echo "$(green_text "╚═════╝ ╚═╝  ╚═╝   ╚═╝     ╚═══╝  ")"

  cat - <<EOF
$(green_text DAYU system is running):
See Pod status: kubectl -n ${NAMESPACE} get pod
See System UI: http://$CLOUD_IP:$FRONTEND_PORT
EOF
}


check_kubectl () {
  kubectl get pod >/dev/null 2>&1
}

check_template() {
    if [ -z "${TEMPLATE-}" ]; then
        echo "$(red_text TEMPLATE) environment variable is not set."
        echo "Please set the $(red_text TEMPLATE) environment variable to the configuration directory path."
        exit 1
    fi

    if [[ "${TEMPLATE}" != /* ]]; then
        TEMPLATE="$(pwd)/${TEMPLATE}"
    fi

    if [[ "${TEMPLATE}" != */ ]]; then
        TEMPLATE="${TEMPLATE}/"
    fi

    if [ ! -d "${TEMPLATE}" ]; then
        echo "The directory specified in $(red_text TEMPLATE) does not exist: ${TEMPLATE}"
        exit 1
    else
        echo "TEMPLATE directory is set to: ${TEMPLATE}"
    fi
}

check_action() {
  action=${ACTION:-start}
  support_action_list="start stop"
  if ! echo "$support_action_list" | grep -w -q "$action"; then
    echo "\`$action\` not in support action list: start/stop!" >&2
    echo "You need to specify it by setting $(red_text ACTION) environment variable when running this script!" >&2
    exit 2
  fi

}

check_official_namespace() {

    local official_namespaces=("default" "kube-node-lease" "kube-public" "kube-system" "kubeedge" "sedna")
    local official_namespaces_str="${official_namespaces[*]}"

    for ns in "${official_namespaces[@]}"; do
        if [[ "$NAMESPACE" == "$ns" ]]; then
            echo "It is not allowed to set the namespace $(red_text "$NAMESPACE") in official namespaces: $official_namespaces_str"
            exit 1
        fi
    done
}

import_config() {
    CONFIG_FILE="$TEMPLATE/base.yaml"
    TMP_FILE="$TEMPLATE/tmp_preprocessed_base.yaml"
    preprocess_yaml "$CONFIG_FILE" "$TMP_FILE"

    if [ ! -f "$TMP_FILE" ]; then
        echo "Configuration file not found at $(red_text "$TMP_FILE")"
        exit 1
    fi

    NAMESPACE=$(yq e '.namespace' "$TMP_FILE")
    LOG_LEVEL=$(yq e '.log-level' "$TMP_FILE")
    BACKEND_SERVICE_ACCOUNT=$(yq e '.backend-rbac.service-account' "$TMP_FILE")
    BACKEND_ROLE=$(yq e '.backend-rbac.role' "$TMP_FILE")
    BACKEND_ROLE_BINDING=$(yq e '.backend-rbac.role-binding' "$TMP_FILE")
    # Cluster-scoped RBAC names are deployment-specific. Otherwise stopping a
    # second namespace could revoke the first namespace's backend access.
    BACKEND_CLUSTER_ROLE="$(yq e '.backend-rbac.cluster-role' "$TMP_FILE")-${NAMESPACE}"
    BACKEND_CLUSTER_ROLE_BINDING="$(yq e '.backend-rbac.cluster-role-binding' "$TMP_FILE")-${NAMESPACE}"
    SUPPORT_API_VERSION=$(yq e '.support-crd-meta.api-version' "$TMP_FILE")
    SUPPORT_KIND=$(yq e '.support-crd-meta.kind' "$TMP_FILE")
    REGISTRY=$(yq e '.default-image-meta.registry' "$TMP_FILE")
    REPOSITORY=$(yq e '.default-image-meta.repository' "$TMP_FILE")
    TAG=$(yq e '.default-image-meta.tag' "$TMP_FILE")
    DEFAULT_FILE_MOUNT_PREFIX=$(yq e '.default-file-mount-prefix' "$TMP_FILE")
    DATASOURCE_USE_SIMULATION=$(yq e '.datasource.use-simulation' "$TMP_FILE")
    DATASOURCE_DATA_ROOT=$(yq e '.datasource.data-root' "$TMP_FILE")
    DATASOURCE_NODE=$(yq e '.datasource.node' "$TMP_FILE")
    DATASOURCE_PLAY_MODE=$(yq e '.datasource.play-mode' "$TMP_FILE")
    SYSTEM_LOG_RETENTION_RECORDS=$(yq e '.log-export.system.retention-records' "$TMP_FILE")
    SYSTEM_LOG_COMPACT_INTERVAL=$(yq e '.log-export.system.compact-interval' "$TMP_FILE")
    rm "$TMP_FILE"

}

preprocess_yaml() {
  local input_file="$1"
  local output_file="$2"

  cp "$input_file" "$output_file"

  local current_dir=$(dirname "$output_file")

  while grep -q '!include' "$output_file"; do
    include_line=$(grep -m1 '!include' "$output_file")
    include_file=$(echo "$include_line" | tr -d '\r' | sed -E 's/.*!include[[:space:]]+["'\'']?([^"'\'']+)["'\'']?.*/\1/')

    local include_path="${current_dir}/${include_file}"
    if [ ! -f "$include_path" ]; then
      echo "Error: Include file '$include_path' not found." >&2
      exit 1
    fi

    include_content=$(sed -e 's/^/  /' "$include_path")

    awk -v include_line="${include_line//\\r/}" -v include_content="$include_content" '
      $0 == include_line { print include_content; found=1; next }
      { print }
    ' "$output_file" > "${output_file}.tmp" && mv "${output_file}.tmp" "$output_file"
  done
}

get_master_details() {
    IFS=$'\n' read -r -d '' -a MASTER_DETAILS < <(kubectl get nodes --selector=node-role.kubernetes.io/master='' --no-headers -o custom-columns="NAME:.metadata.name,INTERNAL_IP:.status.addresses[?(@.type=='InternalIP')].address" | awk '{print $1, $2}' | head -n 1 && printf '\0')

    CLOUD_NODE=${MASTER_DETAILS[0]% *}
    CLOUD_IP=${MASTER_DETAILS[0]##* }

    if [ -z "$CLOUD_NODE" ]; then
        IFS=$'\n' read -r -d '' -a MASTER_DETAILS < <(kubectl get nodes --selector=node-role.kubernetes.io/control-plane='' --no-headers -o custom-columns="NAME:.metadata.name,INTERNAL_IP:.status.addresses[?(@.type=='InternalIP')].address" | awk '{print $1, $2}' | head -n 1 && printf '\0')
        CLOUD_NODE=${MASTER_DETAILS[0]% *}
        CLOUD_IP=${MASTER_DETAILS[0]##* }
    fi

    if [ -z "$CLOUD_NODE" ]; then
        echo "No master/control-plane node found, please check your Kubernetes cluster configuration."
        exit 1
    fi

}

display_config() {
    echo "----------------------------------------"
    echo "        Configuration Imported"
    echo "----------------------------------------"
    echo "  Namespace: $NAMESPACE"
    echo "  Log Level: $LOG_LEVEL"
    echo "  Backend Service Account: $BACKEND_SERVICE_ACCOUNT"
    echo "  Backend Role: $BACKEND_ROLE"
    echo "  Backend Role Binding: $BACKEND_ROLE_BINDING"
    echo "  Backend Cluster Role: $BACKEND_CLUSTER_ROLE"
    echo "  Backend Cluster Role Binding: $BACKEND_CLUSTER_ROLE_BINDING"
    echo "  Support API Version: $SUPPORT_API_VERSION"
    echo "  Support Kind: $SUPPORT_KIND"
    echo "  Registry: $REGISTRY"
    echo "  Repository: $REPOSITORY"
    echo "  Tag: $TAG"
    echo "  Default File Mount Prefix: $DEFAULT_FILE_MOUNT_PREFIX"
    echo "  Datasource Simulation: $DATASOURCE_USE_SIMULATION"
    echo "  Datasource Data Root: $DATASOURCE_DATA_ROOT"
    echo "  Datasource Node: $DATASOURCE_NODE"
    echo "  Datasource Play Mode: $DATASOURCE_PLAY_MODE"
    echo "  Master Node: $CLOUD_NODE"
    echo "  Master Node IP: $CLOUD_IP"
    echo "----------------------------------------"
}

get_service_nodeport() {
  SERVICE_NAME=$1
  NAMESPACE=$2
  NODE_PORT=$(kubectl get svc "$SERVICE_NAME" -n "$NAMESPACE" -o=jsonpath='{.spec.ports[0].nodePort}')
  echo "$NODE_PORT"
}


prepare() {
  echo "Preparing for DAYU system..."
  check_kubectl
  check_action
  check_template
  check_install_yq
  import_config
  get_master_details
  display_config
  check_official_namespace
}

prepare

case "$action" in
  start)
  start_system
    ;;
  stop)
    set +o errexit
    stop_system
    ;;
esac
