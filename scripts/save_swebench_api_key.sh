#!/bin/bash

# Check if API key is provided
if [ -z "$1" ]; then
    echo "Error: No API key provided"
    echo "Usage: $0 <api_key>"
    exit 1
fi

API_KEY="$1"
BASHRC="$HOME/.bashrc"

# Check if SWEBENCH_API_KEY already exists in .bashrc
if grep -q "export SWEBENCH_API_KEY=" "$BASHRC"; then
    echo "SWEBENCH_API_KEY already exists in $BASHRC"
    echo "Updating existing entry..."
    # Use sed to replace the existing line
    sed -i "s|export SWEBENCH_API_KEY=.*|export SWEBENCH_API_KEY=\"$API_KEY\"|" "$BASHRC"
else
    echo "Adding SWEBENCH_API_KEY to $BASHRC"
    # Append to .bashrc
    echo "" >> "$BASHRC"
    echo "# SWE-Bench API Key" >> "$BASHRC"
    echo "export SWEBENCH_API_KEY=\"$API_KEY\"" >> "$BASHRC"
fi

echo "API key saved successfully!"
echo "Sourcing $BASHRC..."

# Source the bashrc file
source ~/.bashrc

echo "Done! SWEBENCH_API_KEY is now set."
echo "Current value: $SWEBENCH_API_KEY"