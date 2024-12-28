#!/bin/bash

show_help() {
    echo "Usage: $0 -f <yaml_file> [-r <services_to_delete>]"
    echo "Delete Kubernetes resources based on the provided YAML file."
    echo ""
    echo "Options:"
    echo "  -f <yaml_file>         Specify the YAML file containing Kubernetes resources."
    echo "  -r <services_to_delete> Specify a list of services to delete (optional)."
    echo "                         If not provided, all services will be deleted."
    echo "                         Example: -r service1 service2"
    echo "  --help                 Display this help message."
    exit 1
}

yaml_file=""
services_to_delete=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -f) yaml_file="$2"; shift 2;;
        -r) shift
            while [[ $# -gt 0 && "$1" != -* ]]; do
                services_to_delete+=("$1")
                shift
            done
            ;;
        --help) show_help;;
        *) show_help;;
    esac
done

# Ensure YAML file is provided
if [ -z "$yaml_file" ]; then
    echo "Error: YAML file not specified."
    show_help
fi

current_doc=""

# Function to process a document
process_document() {
    local doc="$1"
    local namespace=$(echo "$doc" | yq eval '.metadata.namespace // "default"' -)
    local deployment_name=$(echo "$doc" | yq eval '.metadata.name' -)
    local deployment_pos=$(echo "$doc" | yq eval '.spec.serviceConfig.pos' -)
    local svc_name="${deployment_name}-${deployment_pos}"

    # Check if the service should be deleted
    if [ ${#services_to_delete[@]} -eq 0 ] || [[ " ${services_to_delete[@]} " =~ " $deployment_name " ]]; then
        echo "Deleting resources for $deployment_name"

        # 1. Delete SVC
        kubectl delete svc "$svc_name" -n "$namespace" --ignore-not-found

        # 2. Delete Deployments
        deployments=$(kubectl get deploy -n "$namespace" -o json | \
                      jq -r --arg deployment_name "$deployment_name" \
                      '.items[] | select(.metadata.name | startswith($deployment_name)) | .metadata.name')

        for deployment in $deployments; do
            kubectl delete deploy "$deployment" -n "$namespace" --ignore-not-found
        done

        # 3. Delete Custom Resource (jointmultiedgeservice)
        kubectl delete jointmultiedgeservice "$deployment_name" -n "$namespace" --ignore-not-found
    else
        echo "Skipping deletion for $deployment_name"
    fi
}

# Read YAML file and process documents
while IFS='' read -r line || [ -n "$line" ]; do
    if [[ "$line" == "---" ]]; then
        # Process the previous document
        if [ -n "$current_doc" ]; then
            process_document "$current_doc"
        fi
        current_doc=""
    else
        current_doc="$current_doc$line"$'\n'
    fi
done < "$yaml_file"

# Process the last document if it doesn't end with ---
if [ -n "$current_doc" ]; then
    process_document "$current_doc"
fi
