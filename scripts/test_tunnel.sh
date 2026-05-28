#!/bin/bash
# test_tunnel.sh — Run on RunPod to test the two-hop reverse tunnel to babel compute node.
#
# Usage:
#   ./test_tunnel.sh [compute-node]
#   ./test_tunnel.sh babel-p5-20    (default)

set -eo pipefail

PORT=8078
REMOTE_HOST="login.babel.cs.cmu.edu"
REMOTE_USER="srgandhi"
COMPUTE_TARGET="${1:-babel-p5-20}"

echo "=== Testing Two-Hop Reverse Tunnel ==="
echo "Port:         $PORT"
echo "Login node:   $REMOTE_HOST"
echo "Compute node: $COMPUTE_TARGET"
echo ""

# Step 1: Start dummy HTTP server
echo "[1/5] Starting dummy HTTP server on localhost:$PORT..."
python3 -c "
from http.server import HTTPServer, BaseHTTPRequestHandler
class H(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'TUNNEL_OK')
    def log_message(self, *args): pass
HTTPServer(('0.0.0.0', $PORT), H).serve_forever()
" &
DUMMY_PID=$!
sleep 1

# Verify dummy server works locally
if curl -s http://localhost:$PORT | grep -q TUNNEL_OK; then
    echo "  OK: Local server responding"
else
    echo "  FAIL: Local server not responding"
    kill $DUMMY_PID 2>/dev/null
    exit 1
fi

# Step 2: Hop 1 — reverse tunnel to login node
echo ""
echo "[2/5] Setting up hop 1: RunPod -> $REMOTE_HOST ..."
ssh -o StrictHostKeyChecking=no \
    -o ExitOnForwardFailure=yes \
    -c aes128-ctr,aes192-ctr,aes256-ctr \
    -o MACs=hmac-sha2-256,hmac-sha2-512,hmac-sha1 \
    -R ${PORT}:localhost:${PORT} \
    ${REMOTE_USER}@${REMOTE_HOST} \
    -N -f

if [ $? -eq 0 ]; then
    echo "  OK: Hop 1 established"
else
    echo "  FAIL: Hop 1 failed"
    kill $DUMMY_PID 2>/dev/null
    exit 1
fi

# Step 3: Test from login node
echo ""
echo "[3/5] Testing curl from login node..."
RESULT=$(ssh -o StrictHostKeyChecking=no \
    -c aes128-ctr,aes192-ctr,aes256-ctr \
    -o MACs=hmac-sha2-256,hmac-sha2-512,hmac-sha1 \
    ${REMOTE_USER}@${REMOTE_HOST} \
    "curl -s --max-time 5 http://localhost:${PORT}" 2>/dev/null || true)

if echo "$RESULT" | grep -q TUNNEL_OK; then
    echo "  OK: Login node can reach RunPod"
else
    echo "  FAIL: Login node cannot reach RunPod (got: '$RESULT')"
    echo "  Stopping here."
    pkill -f "ssh.*${PORT}.*${REMOTE_HOST}" 2>/dev/null
    kill $DUMMY_PID 2>/dev/null
    exit 1
fi

# Step 4: Direct tunnel to compute node via ProxyJump
echo ""
echo "[4/5] Setting up direct tunnel to $COMPUTE_TARGET (via ProxyJump through login)..."
# Kill hop 1 first — we'll do a single direct tunnel instead
pkill -f "ssh.*${PORT}.*${REMOTE_HOST}" 2>/dev/null
sleep 1

ssh -o StrictHostKeyChecking=no \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    -o ExitOnForwardFailure=yes \
    -J "${REMOTE_USER}@${REMOTE_HOST}" \
    -R "${PORT}:localhost:${PORT}" \
    "${REMOTE_USER}@${COMPUTE_TARGET}" \
    -N -f

if [ $? -eq 0 ]; then
    echo "  OK: Direct tunnel via ProxyJump established"
else
    echo "  FAIL: Could not set up tunnel"
    kill $DUMMY_PID 2>/dev/null
    exit 1
fi
HOP2_PID=""

# Step 5: Test from compute node
echo ""
echo "[5/5] Testing curl from compute node ($COMPUTE_TARGET)..."
RESULT=$(ssh -o StrictHostKeyChecking=no \
    -c aes128-ctr,aes192-ctr,aes256-ctr \
    -o MACs=hmac-sha2-256,hmac-sha2-512,hmac-sha1 \
    ${REMOTE_USER}@${REMOTE_HOST} \
    "ssh -o StrictHostKeyChecking=no ${COMPUTE_TARGET} 'curl -s --max-time 5 http://localhost:${PORT}'" 2>/dev/null || true)

if echo "$RESULT" | grep -q TUNNEL_OK; then
    echo "  OK: Compute node can reach RunPod!"
    echo ""
    echo "=== ALL TESTS PASSED ==="
else
    echo "  FAIL via automated test (got: '$RESULT')"
    echo ""
    echo "=== Automated test inconclusive ==="
fi

echo ""
echo "Server + tunnels still running. Test manually from the compute node:"
echo ""
echo "  curl http://localhost:${PORT}"
echo ""
echo "Press Enter here when done to clean up..."
read -r

# Cleanup
echo "Cleaning up..."
pkill -f "ssh.*${PORT}.*${COMPUTE_TARGET}" 2>/dev/null
pkill -f "ssh.*${PORT}.*${REMOTE_HOST}" 2>/dev/null
# Kill orphaned sshd listeners on compute node and login node
ssh -o StrictHostKeyChecking=no \
    -o ConnectTimeout=5 \
    -c aes128-ctr,aes192-ctr,aes256-ctr \
    -o MACs=hmac-sha2-256,hmac-sha2-512,hmac-sha1 \
    ${REMOTE_USER}@${REMOTE_HOST} \
    "pkill -u ${REMOTE_USER} -f 'sshd.*${PORT}'; ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 ${COMPUTE_TARGET} 'pkill -u ${REMOTE_USER} -f \"sshd.*${PORT}\"'" 2>/dev/null || true
kill $DUMMY_PID 2>/dev/null
echo "Done."
