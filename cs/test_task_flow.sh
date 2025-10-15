#!/bin/bash

# Test script to verify task flow and queue management

echo "Ocean Plastic Forecast - Task Flow Test"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if server is running
echo "Checking server status..."
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo -e "${RED}✗ Server is not running!${NC}"
    echo "Please start the server first:"
    echo "  cd server && uvicorn app.api.main:app --reload"
    exit 1
fi
echo -e "${GREEN}✓ Server is running${NC}"
echo ""

# Function to display queue status
show_queue() {
    echo -e "${YELLOW}Queue Status:${NC}"
    curl -s http://localhost:8000/api/v1/tasks/queue/status | jq '{
        pending: .pending_tasks,
        active: .active_tasks,
        clients: .active_clients,
        capacity: .total_client_capacity,
        config: {
            target: .target_pending_tasks,
            min: .min_pending_tasks,
            max_per_client: .max_tasks_per_client
        }
    }'
    echo ""
}

# Function to show active tasks detail
show_active_tasks() {
    echo -e "${YELLOW}Active Tasks Detail:${NC}"
    curl -s http://localhost:8000/api/v1/admin/debug/active-tasks | jq '{
        total: .total_active_tasks,
        queue_size: .queue_size,
        by_client: (.tasks_by_client | to_entries | map({
            client: .key,
            task_count: (.value | length),
            task_ids: (.value | map(.task_id))
        }))
    }'
    echo ""
}

# Function to show recent results
show_recent_results() {
    echo -e "${YELLOW}Recent Task Results:${NC}"
    curl -s http://localhost:8000/api/v1/tasks/results?limit=5 | jq '.results | map({
        task_id: .task_id,
        client: .client_name,
        particles: .particle_count,
        execution_time: .execution_time,
        simulation_type: .simulation_type,
        user: .user_id
    })'
    echo ""
}

# Function to clear stuck tasks
clear_stuck() {
    echo -e "${YELLOW}Clearing stuck tasks...${NC}"
    curl -s -X POST http://localhost:8000/api/v1/admin/tasks/clear-stuck | jq
    echo ""
}

# Main menu
while true; do
    echo "Available commands:"
    echo "  1) Show queue status"
    echo "  2) Show active tasks detail"
    echo "  3) Show recent results"
    echo "  4) Clear stuck tasks"
    echo "  5) Watch queue status (live)"
    echo "  6) Create test batch"
    echo "  7) Run full diagnostic"
    echo "  q) Quit"
    echo ""
    read -p "Enter choice: " choice
    echo ""
    
    case $choice in
        1)
            show_queue
            ;;
        2)
            show_active_tasks
            ;;
        3)
            show_recent_results
            ;;
        4)
            clear_stuck
            show_queue
            ;;
        5)
            echo "Watching queue status (Ctrl+C to stop)..."
            echo ""
            watch -n 2 'curl -s http://localhost:8000/api/v1/tasks/queue/status | jq "{pending: .pending_tasks, active: .active_tasks, clients: .active_clients}"'
            ;;
        6)
            echo -e "${YELLOW}Creating test batch...${NC}"
            curl -s -X POST http://localhost:8000/api/v1/simulations/test-batch | jq
            echo ""
            show_queue
            ;;
        7)
            echo -e "${GREEN}Running full diagnostic...${NC}"
            echo ""
            show_queue
            show_active_tasks
            show_recent_results
            
            echo -e "${YELLOW}System Stats:${NC}"
            curl -s http://localhost:8000/api/v1/admin/stats | jq
            echo ""
            ;;
        q|Q)
            echo "Exiting..."
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid choice${NC}"
            echo ""
            ;;
    esac
done

