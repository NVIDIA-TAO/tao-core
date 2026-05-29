#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Build script for FTMS Docker image

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${IMAGE_NAME:-nvcr.io/nvstaging/tao/tao-ftms}"
IMAGE_TAG="${IMAGE_TAG:-dev}"

echo "Building FTMS Docker image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "Working directory: ${SCRIPT_DIR}"

cd "${SCRIPT_DIR}"

# Build the Docker image
docker build \
    -f Dockerfile \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" \
    .

echo ""
echo "✅ Build complete!"
echo "To run the container:"
echo "  docker run -p 8000:8000 ${IMAGE_NAME}:${IMAGE_TAG}"
echo ""
echo "To push to registry:"
echo "  docker push ${IMAGE_NAME}:${IMAGE_TAG}"

