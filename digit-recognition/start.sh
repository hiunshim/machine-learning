#!/bin/zsh

# Explicitly source .zshrc to make build_venv available
source ~/.zshrc

# Call the build_venv function
build_venv

# Train and save the model
python -c "from model import train_and_save_model; train_and_save_model()"

# Download a random test digit
python download_random_test_digit.py

# Start the API server
uvicorn main:app --reload

