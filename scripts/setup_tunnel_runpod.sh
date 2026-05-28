#!/bin/bash
# setup_tunnel_runpod.sh — Run on a fresh RunPod instance.
#
# 1. Generates an SSH key (if none exists)
# 2. Copies it to the babel login node (one-time password prompt)
# 3. Establishes a reverse SSH tunnel: RunPod:PORT -> babel compute node localhost:PORT
#
# Usage:
#   ./setup_tunnel_runpod.sh [options]
#
# Options:
#   --port <port>           Local port to forward (default: 8078)
#   --remote-port <port>    Port on compute node (default: same as --port)
#   --remote-host <host>    Babel login node (default: login.babel.cs.cmu.edu)
#   --remote-user <user>    SSH user (default: srgandhi)
#   --compute-node <host>   Babel compute node (default: babel-p5-20)
#   --keys-only             Only set up SSH keys, don't start the tunnel
#   --tunnel-only           Skip key setup, just start the tunnel

set -eo pipefail

# Defaults
PORT=8078
REMOTE_PORT=""
REMOTE_HOST="login.babel.cs.cmu.edu"
REMOTE_USER="srgandhi"
COMPUTE_TARGET="babel-p5-20"
KEYS_ONLY=false
TUNNEL_ONLY=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --port)           PORT="$2";           shift 2 ;;
    --remote-port)    REMOTE_PORT="$2";    shift 2 ;;
    --remote-host)    REMOTE_HOST="$2";    shift 2 ;;
    --remote-user)    REMOTE_USER="$2";    shift 2 ;;
    --compute-node)   COMPUTE_TARGET="$2"; shift 2 ;;
    --keys-only)      KEYS_ONLY=true;      shift ;;
    --tunnel-only)    TUNNEL_ONLY=true;    shift ;;
    *)                echo "Unknown option: $1"; exit 1 ;;
  esac
done

[ -z "$REMOTE_PORT" ] && REMOTE_PORT=$PORT

KEY_FILE="$HOME/.ssh/id_ed25519"

# ── Step 1: SSH key setup ─────────────────────────────────────────────────────
setup_keys() {
    echo "=== SSH Key Setup ==="

    # Ensure .ssh dir exists with correct permissions
    mkdir -p "$HOME/.ssh"
    chmod 700 "$HOME/.ssh"

    # Generate key if it doesn't exist
    if [ -f "$KEY_FILE" ]; then
        echo "SSH key already exists: $KEY_FILE"
    else
        echo "Generating new ed25519 key..."
        ssh-keygen -t ed25519 -f "$KEY_FILE" -N "" -C "runpod-$(hostname)-$(date +%Y%m%d)"
        echo "Key generated."
    fi

    echo ""
    echo "Copying key to ${REMOTE_USER}@${REMOTE_HOST}..."
    echo "You will be prompted for your babel password (one time only)."
    echo ""
    ssh-copy-id -o StrictHostKeyChecking=no "${REMOTE_USER}@${REMOTE_HOST}"

    # Verify passwordless login works
    echo ""
    echo "Verifying passwordless SSH..."
    if ssh -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=10 \
         "${REMOTE_USER}@${REMOTE_HOST}" "echo SSH_OK" 2>/dev/null | grep -q SSH_OK; then
        echo "Passwordless SSH to ${REMOTE_HOST} is working."
    else
        echo "ERROR: Passwordless SSH verification failed."
        echo "Try manually: ssh ${REMOTE_USER}@${REMOTE_HOST}"
        exit 1
    fi

    # Verify we can reach the compute node through the login node
    echo "Verifying access to compute node ${COMPUTE_TARGET}..."
    if ssh -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=10 \
         -J "${REMOTE_USER}@${REMOTE_HOST}" \
         "${REMOTE_USER}@${COMPUTE_TARGET}" "echo SSH_OK" 2>/dev/null | grep -q SSH_OK; then
        echo "Access to ${COMPUTE_TARGET} via ProxyJump is working."
    else
        echo "WARNING: Could not reach ${COMPUTE_TARGET} via ProxyJump."
        echo "The compute node may not be running. Tunnel setup may fail."
    fi

    echo ""
    echo "=== Key Setup Complete ==="
}

# ── Step 2: Reverse SSH tunnel ────────────────────────────────────────────────
start_tunnel() {
    echo ""
    echo "=== Reverse SSH Tunnel ==="
    echo "RunPod localhost:${PORT}  -->  ${COMPUTE_TARGET} localhost:${REMOTE_PORT}"
    echo "Via ProxyJump through ${REMOTE_HOST}"
    echo ""

    # Kill any existing tunnel on this port
    pkill -f "ssh.*-R.*${REMOTE_PORT}:localhost:${PORT}.*${COMPUTE_TARGET}" 2>/dev/null || true
    sleep 1

    ssh -o StrictHostKeyChecking=no \
        -o ServerAliveInterval=30 \
        -o ServerAliveCountMax=3 \
        -o ExitOnForwardFailure=yes \
        -J "${REMOTE_USER}@${REMOTE_HOST}" \
        -R "${REMOTE_PORT}:localhost:${PORT}" \
        "${REMOTE_USER}@${COMPUTE_TARGET}" \
        -N -f

    if [ $? -eq 0 ]; then
        echo "=== Tunnel Active ==="
        echo "From ${COMPUTE_TARGET}: curl http://localhost:${REMOTE_PORT}/v1"
        echo ""
        echo "To stop: pkill -f 'ssh.*${REMOTE_PORT}.*${COMPUTE_TARGET}'"
        echo "======================"
    else
        echo "ERROR: Tunnel setup failed."
        echo "Check that ${COMPUTE_TARGET} is reachable and port ${REMOTE_PORT} is free."
        exit 1
    fi
}

# ── Main ──────────────────────────────────────────────────────────────────────
if [ "$TUNNEL_ONLY" = true ]; then
    start_tunnel
elif [ "$KEYS_ONLY" = true ]; then
    setup_keys
else
    setup_keys
    start_tunnel
fi
