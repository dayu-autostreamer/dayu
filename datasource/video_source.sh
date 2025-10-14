#!/bin/bash

# Track whether this script started mediamtx and its PID so we can stop it on exit
STARTED_MEDIAMTX=0
MEDIAMTX_PID=""

cleanup() {
    # Stop mediamtx only if we started it
    if [[ "$STARTED_MEDIAMTX" -eq 1 && -n "$MEDIAMTX_PID" ]]; then
        if kill -0 "$MEDIAMTX_PID" 2>/dev/null; then
            echo "Stopping mediamtx (pid=$MEDIAMTX_PID)..."
            kill "$MEDIAMTX_PID" 2>/dev/null || true
            # Wait briefly, then force kill if still alive
            for _ in {1..20}; do
                if ! kill -0 "$MEDIAMTX_PID" 2>/dev/null; then
                    break
                fi
                sleep 0.2
            done
            if kill -0 "$MEDIAMTX_PID" 2>/dev/null; then
                echo "Force killing mediamtx (pid=$MEDIAMTX_PID)..."
                kill -9 "$MEDIAMTX_PID" 2>/dev/null || true
            fi
        fi
    fi
    # Also try to stop any background jobs started by this script (if any)
    bg_pids=$(jobs -p)
    if [[ -n "$bg_pids" ]]; then
        kill $bg_pids 2>/dev/null || true
    fi
}

cleanup_and_exit() {
    # Disarm traps to avoid recursion when calling exit
    trap - INT TERM EXIT
    cleanup
    exit 0
}

# On SIGINT/SIGTERM, cleanup and exit to stop the streaming loop immediately
trap cleanup_and_exit INT TERM

function show_help() {
    echo "Usage: $0 --root <folder> --address <address>"
    echo
    echo "This script continuously streams all video files in the input folder to an RTSP server."
    echo
    echo "Arguments:"
    echo "  --root <folder>:    Path to the folder containing video files."
    echo "  --address <address>: Specific address to append to 'rtsp://127.0.0.1/'."
    echo "  --play_mode <mode>: cycle or non-cycle play mode"
    echo
    echo "Example:"
    echo "  $0 --root /path/to/your/video/folder --address your_specific_address"
    exit 1
}


input_folder=""
rtsp_address=""
play_mode=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --root) input_folder="$2"; shift ;;
        --address) rtsp_address="$2"; shift ;;
        --play_mode) play_mode="$2"; shift ;;
        -h|--help) show_help ;;
        *) echo "Unknown parameter passed: $1"; show_help ;;
    esac
    shift
done

if [[ -z "$input_folder" || -z "$rtsp_address" || -z "$play_mode" ]]; then
    echo "Error: --root, --address and --play_mode are required."
    show_help
fi

if [[ ! -d "$input_folder" ]]; then
    echo "Error: The specified root path '$input_folder' is not a directory or does not exist."
    exit 1
fi

if ! pgrep -x "mediamtx" > /dev/null; then
    echo "Starting mediamtx server..."
    # Start mediamtx in background and capture its PID for cleanup
    "$RTSP_PATH"/mediamtx "$RTSP_PATH"/mediamtx.yml > /dev/null 2>&1 &
    MEDIAMTX_PID=$!
    STARTED_MEDIAMTX=1
    sleep 4
else
    echo "mediamtx server already running."
fi

rtsp_url="$rtsp_address"

# Function to stream video files once or cycle based on play_mode
stream_videos() {
    local play_mode=$1
    while true; do
        # Get all video files in the folder
        video_files=("$input_folder"/*)

        # Check if there are any video files in the directory
        if [[ ${#video_files[@]} -eq 0 ]]; then
            echo "No video files found in the folder '$input_folder'."
            break
        fi

        for video_file in "${video_files[@]}"; do
            if [[ -f "$video_file" ]]; then
                echo "Streaming $video_file to $rtsp_url"
                ffmpeg -re -i "$video_file" -c copy -b:v 3000k -f rtsp -rtsp_transport tcp "$rtsp_url"
            fi
        done

        # If play_mode is "non-cycle", break after playing once
        if [[ "$play_mode" == "non-cycle" ]]; then
            echo "Stream video in non-cycle mode and end."
            break
        fi
    done
}

stream_videos "$play_mode"