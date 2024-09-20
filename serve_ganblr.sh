#!/bin/bash

# Step 1: Setup environment (virtualenv or conda)
# For virtualenv (example):
# python3 -m venv ganblr_env
# source ganblr_env/bin/activate

# For conda (example):
conda create --name ganblr_env python=3.9.19 -y
conda activate ganblr_env

# Step 2: Install libraries in precise order
echo "Installing required libraries..."
pip install pyitlib
pip install tensorflow
pip install pgmpy
pip install sdv
pip install scikit-learn==1.0
pip install fastapi uvicorn

# Step 3: Serve the model API
uvicorn ganblr_api:app --reload --host 0.0.0.0 --port 8000