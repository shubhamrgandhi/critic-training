#!/bin/bash
# Duration threshold in seconds (5 hours = 18000 seconds)
THRESHOLD=18000
# Check interval (run every 1 hour)
CHECK_INTERVAL=1000
echo "Starting SWE-bench container cleaner..."
echo "Will kill containers with 'swebench/sweb.eval' or 'minisweagent' running > 5 hours"
echo "Checking every 1 hour..."
while true; do
    echo ""
    echo "=== Checking at $(date) ==="
    
    # Get all containers matching either pattern (check both image AND name)
    containers=$(docker ps --format '{{.ID}}\t{{.Image}}\t{{.Names}}\t{{.RunningFor}}' | grep -E 'swebench/sweb.eval|minisweagent')
    
    if [ -z "$containers" ]; then
        echo "No swebench/sweb.eval or minisweagent containers found"
    else
        echo "$containers" | while IFS=$'\t' read -r container_id image name running_for; do
            # Get container status in seconds using docker inspect
            created=$(docker inspect --format='{{.State.StartedAt}}' "$container_id")
            created_seconds=$(date -d "$created" +%s)
            current_seconds=$(date +%s)
            running_seconds=$((current_seconds - created_seconds))
            
            echo "Container $container_id ($name / $image): running for $running_seconds seconds (~$((running_seconds/3600)) hours)"
            
            if [ "$running_seconds" -gt "$THRESHOLD" ]; then
                echo "  → Killing container $container_id (running > 5 hours)"
                docker kill "$container_id"
            else
                echo "  → Keeping container (running < 5 hours)"
            fi
        done
    fi
    
    echo ""
    echo "Next check in 1 hour..."
    sleep $CHECK_INTERVAL
done