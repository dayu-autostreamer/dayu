#!/usr/bin/env bash

dayu::util::install_yq(){
    local REQUIRED_YQ_VERSION="4.34.1"
    version_ge() {
        [ "$(printf '%s\n' "$1" "$2" | sort -V | head -n1)" = "$2" ]
    }


    if command -v yq &>/dev/null; then
        local CURRENT_YQ_VERSION
        CURRENT_YQ_VERSION=$(yq --version | awk '{print $3}' | tr -d 'v')
        echo "Current yq version：$CURRENT_YQ_VERSION"
        if version_ge "$CURRENT_YQ_VERSION" "$REQUIRED_YQ_VERSION"; then
            return 0
        else
            echo "Current yq version：$CURRENT_YQ_VERSION(required >= $REQUIRED_YQ_VERSION),updating yq..."
        fi
    else
        echo "yq could not be found, installing..."
    fi

    local YQ_URL
    YQ_URL="https://github.com/mikefarah/yq/releases/download/v$REQUIRED_YQ_VERSION/yq_linux_amd64"

    wget -O /usr/local/bin/yq "$YQ_URL"
    if [[ $? -ne 0 ]]; then
        echo "Downloading yq failed. Please try to install yq manually" >&2
        return 1
    fi

    sudo chmod +x /usr/local/bin/yq

    if ! command -v yq &>/dev/null; then
        echo "Install yq failed. Please try to install yq manually" >&2
        return 1
    fi
}
