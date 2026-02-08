#!/bin/bash
set -e

echo "Pulling data from DVC remote..."
dvc pull

echo "Running SARIMA training..."
python src/training/train_sarima.py