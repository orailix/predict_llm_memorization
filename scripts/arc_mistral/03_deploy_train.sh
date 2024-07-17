#!/bin/bash

source ~/.bashrc
cd /gpfswork/rech/yfw/upp42qa/grokking_llm
conda activate expe

# Function to handle SIGUSR1 signal
sigusr1_handler() {
    echo "Received SIGUSR1 signal"
    # Pass the signal to the Python script
    kill -USR1 $PYTHON_SCRIPT_PID;
    wait $PYTHON_SCRIPT_PID;
}

# Register the sigusr1_handler function to handle SIGUSR1 signal
trap 'sigusr1_handler' USR1 SIGINT SIGTERM

# Prepare deploy
python -u -m grokking_llm deploy-gpu \
    --config=H5gQZhfMsLE5_NTX9hR7TA \
    --training &
PYTHON_SCRIPT_PID=$!

# Wait for the Python script to finish
wait $PYTHON_SCRIPT_PID;