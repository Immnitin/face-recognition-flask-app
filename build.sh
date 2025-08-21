#!/usr/bin/env bash
# Exit on error
set -o errexit

# Install system dependencies
apt-get update && apt-get install -y build-essential cmake libsm6 libxext6 libxrender-dev

# Install Python dependencies
pip install -r requirements.txt