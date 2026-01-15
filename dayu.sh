#!/bin/bash

set -o errexit
set -o nounset
set -o pipefail


check_install_yq() {
    if ! command -v yq &> /dev/null; then
        echo "yq could not be found, installing..."
        wget -O /usr/local/bin/yq https://github.com/mikefarah/yq/releases/download/v4.6.1/yq_linux_amd64
        chmod +x /usr/local/bin/yq
        echo "yq installed successfully."
    fi
}



NO_COLOR='\033[0m'
RED='\033[0;31m'
GREEN='\033[0;32m'
green_text() {
  echo -ne "$GREEN$@$NO_COLOR"
}

red_text() {
  echo -ne "$RED$@$NO_COLOR"
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
    echo "$(green_text [DAYU]) Creating service account and cluster role binding..."
    kubectl -n "$NAMESPACE" create serviceaccount "$SERVICE_ACCOUNT"
    if ! kubectl get clusterrolebinding "$CLUSTER_ROLE_BINDING" > /dev/null 2>&1; then
        kubectl create clusterrolebinding "$CLUSTER_ROLE_BINDING" --clusterrole=cluster-admin --serviceaccount="$NAMESPACE:$SERVICE_ACCOUNT"
    else
        PATCH_JSON="[{\"op\": \"add\", \"path\": \"/subjects/-\", \"value\": {\"kind\": \"ServiceAccount\", \"name\": \"$SERVICE_ACCOUNT\", \"namespace\": \"$NAMESPACE\"}}]"
        kubectl patch clusterrolebinding "$CLUSTER_ROLE_BINDING" --type='json' -p="$PATCH_JSON"
    fi
}

create_redis() {
  echo "$(green_text [DAYU]) Creating redis ..."
      kubectl -n "$NAMESPACE" apply -f - <<EOF
apiVersion: $API_VERSION
kind: $KIND
metadata:
  name: redis
  namespace: $NAMESPACE
spec:
  cloudWorker:
    logLevel:
      level: "DEBUG"
    template:
      spec:
        containers:
          - image: $REGISTRY/redis:latest
            imagePullPolicy: Always
            name: redis
            ports:
              - containerPort: 6379
        dnsPolicy: ClusterFirstWithHostNet
        nodeName: $CLOUD_NODE
        serviceAccountName: $SERVICE_ACCOUNT
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
apiVersion: $API_VERSION
kind: $KIND
metadata:
  name: datasource
  namespace: $NAMESPACE
spec:
  edgeWorker:
    - file:
        paths:
          - $DATASOURCE_DATA_ROOT
          - $DEFAULT_FILE_MOUNT_PREFIX/temp/
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
                - name: KUBERNETES_SERVICE_HOST
                  value: "$KUBERNETES_SERVICE_HOST"
                - name: KUBERNETES_SERVICE_PORT
                  value: "$KUBERNETES_SERVICE_PORT"
                - name: GUNICORN_PORT
                  value: "8000"
                - name: KUBE_CACHE_TTL
                  value: "$KUBE_CACHE_TTL"
                - name: FILE_PREFIX
                  value: "$DATASOURCE_DATA_ROOT"
              image: $REGISTRY/$REPOSITORY/datasource:$TAG
              imagePullPolicy: Always
              name: datasource
              ports:
                - containerPort: 8000
          dnsPolicy: ClusterFirstWithHostNet
          nodeName: $DATASOURCE_NODE
          serviceAccountName: $SERVICE_ACCOUNT
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
apiVersion: $API_VERSION
kind: $KIND
metadata:
  name: backend
  namespace: $NAMESPACE
spec:
  cloudWorker:
    file:
      paths:
      - $TEMPLATE
      - $DEFAULT_FILE_MOUNT_PREFIX/temp/
    logLevel:
      level: "DEBUG"
    template:
      spec:
        containers:
          - env:
            - name: GUNICORN_PORT
              value: "8000"
            - name: KUBE_CACHE_TTL
              value: "$KUBE_CACHE_TTL"
            image: $REGISTRY/$REPOSITORY/backend:$TAG
            imagePullPolicy: Always
            name: backend
            ports:
              - containerPort: 8000
        dnsPolicy: ClusterFirstWithHostNet
        nodeName: $CLOUD_NODE
        serviceAccountName: $SERVICE_ACCOUNT
  serviceConfig:
    port: 8000
    pos: cloud
    targetPort: 8000
EOF


}

create_frontend() {
  echo "$(green_text [DAYU]) Creating frontend ..."
  BACKEND_PORT=$(get_service_nodeport "backend-cloud" "$NAMESPACE")
      kubectl -n "$NAMESPACE" apply -f - <<EOF
apiVersion: $API_VERSION
kind: $KIND
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
              value: 'http://$CLOUD_IP:$BACKEND_PORT'
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
        nodeName: $CLOUD_NODE
        serviceAccountName: $SERVICE_ACCOUNT
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
  INDEX=$(kubectl get clusterrolebinding "$CLUSTER_ROLE_BINDING" -o json | jq '.subjects | to_entries | map(select(.value.kind == "ServiceAccount" and .value.name == "'"$SERVICE_ACCOUNT"'" and .value.namespace == "'"$NAMESPACE"'")) | .[0].key')

  if [[ $INDEX != "null" ]]; then
      PATCH_JSON="[{\"op\": \"remove\", \"path\": \"/subjects/$INDEX\"}]"
      kubectl patch clusterrolebinding "$CLUSTER_ROLE_BINDING" --type='json' -p="$PATCH_JSON"
      echo "$(green_text [DAYU]) Delete service account $SERVICE_ACCOUNT from $CLUSTER_ROLE_BINDING."

      SUBJECTS_JSON=$(kubectl get clusterrolebinding "worker-admin-binding" -o json)
      COUNT=$(echo "$SUBJECTS_JSON" | jq '.subjects | if . then [.[] | select(.kind == "ServiceAccount")] | length else 0 end')
      if [[ "$COUNT" -eq 0 ]]; then
        kubectl delete clusterrolebinding "$CLUSTER_ROLE_BINDING"
        echo "$(green_text [DAYU]) Delete clusterrolebinding $CLUSTER_ROLE_BINDING since no other service accounts are left."
    fi
  fi

}


stop_system() {
    echo "$(green_text [DAYU]) Stopping and cleaning up DAYU system in namespace $NAMESPACE..."
    if check_namespace_existence; then
        delete_service_account

        echo "$(green_text [DAYU]) Delete resources in namespace $NAMESPACE"
        kubectl delete pods,services,deployments,replicasets,statefulsets,horizontalpodautoscalers,jobs,cronjobs,configmaps,secrets --all -n "$NAMESPACE"

        echo "Waiting for all resources to be deleted..."
        while kubectl get all -n "$NAMESPACE" | grep -q .; do
            sleep 1
        done

        echo "$(green_text [DAYU]) Delete namespace $NAMESPACE"
        kubectl delete namespace "$NAMESPACE"

        echo "$(green_text DAYU system stop successfully.)"

    else
        echo "Namespace $(red_text "$NAMESPACE") does not exist. No need to clean up resources."
        exit 1
    fi
}

stop_system() {
    local ns="${NAMESPACE}"
    local svc_wait="${SVC_WAIT_SEC:-120}"
    local mesh_wait="${MESH_WAIT_SEC:-180}"
    local pod_wait="${POD_WAIT_SEC:-180}"
    local ns_wait="${NS_WAIT_SEC:-180}"

    echo "$(green_text [DAYU]) Stopping DAYU system in namespace ${ns}..."

    # ---------------- helper: wait until a namespaced resource kind becomes empty ----------------
    _wait_empty() {
        local kind="$1"
        local namespace="$2"
        local timeout="$3"

        local start_ts
        start_ts="$(date +%s)"

        while true; do
            local out cnt
            out="$(kubectl get "${kind}" -n "${namespace}" --no-headers 2>/dev/null || true)"
            cnt="$(printf '%s' "${out}" | wc -l | tr -d ' ')"

            if [[ "${cnt}" == "0" ]]; then
                return 0
            fi

            if (( $(date +%s) - start_ts > timeout )); then
                echo "$(red_text [DAYU]) ERROR: timeout waiting '${kind}' in '${namespace}' to be empty (still ${cnt})"
                return 1
            fi
            sleep 2
        done
    }

    # ---------------- helper: list all edgemesh-agent pods (ns/pod) ----------------
    _list_edgemesh_pods() {
        kubectl get pods -A --no-headers \
            -o custom-columns=NS:.metadata.namespace,NAME:.metadata.name 2>/dev/null \
          | awk '$2 ~ /^edgemesh-agent/ {print $1"/"$2}'
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

            # Exact match comment: --comment "namespace/xxx"
            if kubectl exec -n "${pns}" "${pname}" -- sh -c \
              "iptables -t nat -S 2>/dev/null | grep -F -- \"--comment \\\"${namespace}/\" >/dev/null" \
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
            echo "$(green_text [DAYU]) No edgemesh-agent pods found, skip EdgeMesh rule wait."
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
                echo "$(red_text [DAYU]) ERROR: EdgeMesh iptables rules for '${namespace}' still exist after ${timeout}s"
                return 1
            fi
            sleep 2
        done
    }


    echo "$(green_text [DAYU]) (0/5) Delete DAYU custom resources ($KIND) to stop controllers from recreating Services..."
    kubectl delete "${KIND}" -n "${ns}" --all --ignore-not-found=true 2>/dev/null || true

    echo "$(green_text [DAYU]) (1/5) Delete Services/Endpoints..."
    kubectl delete svc -n "${ns}" --all --ignore-not-found=true 2>/dev/null || true
    kubectl delete endpoints -n "${ns}" --all --ignore-not-found=true 2>/dev/null || true
    kubectl delete endpointslices -n "${ns}" --all --ignore-not-found=true 2>/dev/null || true
    kubectl delete ingress -n "${ns}" --all --ignore-not-found=true 2>/dev/null || true

    echo "$(green_text [DAYU]) Waiting service/endpoints to be empty..."
    _wait_empty svc "${ns}" "${svc_wait}" || true
    _wait_empty endpoints "${ns}" "${svc_wait}" || true
    _wait_empty endpointslices "${ns}" "${svc_wait}" || true

    echo "$(green_text [DAYU]) (2/5) Waiting EdgeMesh to remove iptables rules for '${ns}'..."
    if ! _wait_edgemesh_rules_gone "${ns}" "${mesh_wait}"; then
        echo "$(red_text [DAYU]) WARN: EdgeMesh didn't cleanup rules in time."
    fi

    echo "$(green_text [DAYU]) (3/5) Delete workloads and remaining resources in namespace '${ns}'..."
    kubectl delete deploy,sts,ds,rs,po,job,cronjob,hpa -n "${ns}" --all --ignore-not-found=true 2>/dev/null || true
    kubectl delete cm,secret -n "${ns}" --all --ignore-not-found=true 2>/dev/null || true

    echo "$(green_text [DAYU]) Waiting pods to be gone..."
    _wait_empty pods "${ns}" "${pod_wait}" || true

    echo "$(green_text [DAYU]) (4/5) Delete service account binding..."
    delete_service_account || true
    kubectl delete sa -n "${ns}" --all --ignore-not-found=true 2>/dev/null || true
    kubectl delete role,rolebinding -n "${ns}" --all --ignore-not-found=true 2>/dev/null || true

    echo "$(green_text [DAYU]) (5/5) Delete namespace '${ns}'..."
    kubectl delete namespace "${ns}" --ignore-not-found=true --wait=true --timeout="${ns_wait}s" 2>/dev/null || true

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
    SERVICE_ACCOUNT=$(yq e '.pod-permission.service-account' "$TMP_FILE")
    CLUSTER_ROLE_BINDING=$(yq e '.pod-permission.cluster-role-binding' "$TMP_FILE")
    API_VERSION=$(yq e '.crd-meta.api-version' "$TMP_FILE")
    KIND=$(yq e '.crd-meta.kind' "$TMP_FILE")
    KUBE_CACHE_TTL=$(yq e '.kube-cache-ttl' "$TMP_FILE")
    REGISTRY=$(yq e '.default-image-meta.registry' "$TMP_FILE")
    REPOSITORY=$(yq e '.default-image-meta.repository' "$TMP_FILE")
    TAG=$(yq e '.default-image-meta.tag' "$TMP_FILE")
    DEFAULT_FILE_MOUNT_PREFIX=$(yq e '.default-file-mount-prefix' "$TMP_FILE")
    DATASOURCE_USE_SIMULATION=$(yq e '.datasource.use-simulation' "$TMP_FILE")
    DATASOURCE_DATA_ROOT=$(yq e '.datasource.data-root' "$TMP_FILE")
    DATASOURCE_NODE=$(yq e '.datasource.node' "$TMP_FILE")
    DATASOURCE_PLAY_MODE=$(yq e '.datasource.play-mode' "$TMP_FILE")

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

get_kubernetes_service_endpoint() {
    local namespace=default

    local api_endpoint=$(kubectl get ep kubernetes --namespace "$namespace" -o jsonpath='{.subsets[0].addresses[0].ip}')
    local api_port=$(kubectl get ep kubernetes --namespace "$namespace" -o jsonpath='{.subsets[0].ports[0].port}')

    if [[ -z "$api_endpoint" || -z "$api_port" ]]; then
        echo "Failed to retrieve Kubernetes $(red_text endpoint) information. Try 'kubectl get ep kubernetes --n $namespace' to debug."
        return 1
    fi

    KUBERNETES_SERVICE_HOST=$api_endpoint
    KUBERNETES_SERVICE_PORT=$api_port

}

display_config() {
    echo "----------------------------------------"
    echo "        Configuration Imported"
    echo "----------------------------------------"
    echo "  Namespace: $NAMESPACE"
    echo "  Log Level: $LOG_LEVEL"
    echo "  Service Account: $SERVICE_ACCOUNT"
    echo "  Cluster Role Binding: $CLUSTER_ROLE_BINDING"
    echo "  API Version: $API_VERSION"
    echo "  Kind: $KIND"
    echo "  Kube Cache TTL: $KUBE_CACHE_TTL"
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
    echo "  Kubernetes Service Host: $KUBERNETES_SERVICE_HOST"
    echo "  Kubernetes Service Port: $KUBERNETES_SERVICE_PORT"
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
  get_kubernetes_service_endpoint
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
