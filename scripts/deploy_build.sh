#!/bin/bash

# Note: this script builds for arm64 and is tested on NVIDIA Jetson Orin NX 16GB

# Exit immediately if a command exits with a non-zero status
set -e

# Find the script's path
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

if [ "${CLEAN_BUILD:-false}" = "true" ]; then
  rm -rf "${SCRIPT_DIR}/../github_clones"
  docker rmi aircraft-image:latest || true
  docker builder prune -f # If CLEAN_BUILD is "true", rebuild everything from scratch
fi

BUILD_DOCKER=true
if [ "${CLONE_ONLY:-false}" = "true" ]; then
  BUILD_DOCKER=false # If CLONE_ONLY is "true", disable the build steps
fi

# Create a folder (ignored by git) to clone GitHub repos
CLONE_DIR="${SCRIPT_DIR}/../github_clones"
mkdir -p "$CLONE_DIR"

REPOS=( # Format: "URL;BRANCH;LOCAL_DIR_NAME"
  # Aircraft image
  # Add repos for PDS-Swarm
  "https://github.com/microsoft/onnxruntime.git;v1.22.1;onnxruntime" # Only for the deployment build
  "https://github.com/PRBonn/kiss-icp.git;main;kiss-icp"
)

for repo_info in "${REPOS[@]}"; do
  IFS=';' read -r url branch dir <<< "$repo_info" # Split the string into URL, BRANCH, and DIR
  TARGET_DIR="${CLONE_DIR}/${dir}"
  if [ -d "$TARGET_DIR" ]; then
    cd "$TARGET_DIR"
    BRANCH=$(git branch --show-current)
    TAGS=$(git tag --points-at HEAD)
    echo "There is a clone of ${dir} on branch: ${BRANCH}, tags: [${TAGS}]"
    # The script does not automatically pull changes for already cloned repos (as they should be on fixed tags)
    # git pull
    # git submodule update --init --recursive --depth 1
    cd "$CLONE_DIR"
  else
    echo "Clone not found, cloning ${dir}..."
    TEMP_DIR="${TARGET_DIR}_temp"     
    rm -rf "$TEMP_DIR" # Clean up any failed clone from a previous run   
    git clone --depth 1 --branch "$branch" --recursive "$url" "$TEMP_DIR" && mv "$TEMP_DIR" "$TARGET_DIR"
  fi
done

if [ "$BUILD_DOCKER" = "true" ]; then
  # Make sure AAS's Git LFS simulation resources are pulled (although not strictly necessary for deployment)
  git lfs install
  git lfs pull

  # The first build takes ~1h (mostly to build onnxruntime-gpu from source) and creates an 18GB image
  docker build -t aircraft-image -f "${SCRIPT_DIR}/docker/Dockerfile.aircraft" "${SCRIPT_DIR}/.."
else
  echo -e "Skipping Docker build"
fi