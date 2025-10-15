# Task Queue Fix - Continuous Task Processing

## Problem
The client was able to complete 6 tasks but then couldn't get more tasks from the server. The server thought the client still had 5 active tasks assigned, even though they had completed.

## Root Cause
The task completion messages weren't being properly processed, leaving "ghost" tasks in the `active_tasks` dictionary on the server side. This caused the server to refuse assigning more tasks since it thought the client had reached its limit.

## Changes Made

### 1. Enhanced Logging (`task_manager.py`)

**In `assign_task()`:**
- Added detailed logging showing client task count vs limit
- Shows which specific tasks are active for a client
- Logs queue size changes
- Triggers batch creation if queue is empty

**In `complete_task()`:**
- Added logging to show when task isn't found in active_tasks
- Shows which client was expected vs which reported completion
- Logs active task count after completion
- More detailed error messages

**In `_check_and_create_tasks()`:**
- Clearer logging about queue status
- Shows when and why batches are created
- Indicates successful batch creation with new queue size

### 2. WebSocket Message Handling (`main.py`)

**In `task_completed` handler:**
- Added logging when completion message is received
- Logs execution time
- Shows success/failure of completion
- Logs database storage
- Confirms acknowledgment sent

### 3. Debug Endpoint (`admin.py`)

Added new endpoint: `GET /api/v1/admin/debug/active-tasks`

Returns detailed information about:
- All active tasks with their assignments
- Tasks grouped by client
- Task age (how long they've been active)
- Current queue size

## How It Works Now

### Task Lifecycle:
1. **Server creates batch** → 10 tasks added to queue
2. **Client requests task** → Server assigns from queue, adds to `active_tasks`
3. **Client completes task** → Sends `task_completed` via WebSocket
4. **Server receives completion** → Removes from `active_tasks`, stores result
5. **Queue check triggered** → If queue < 5, creates new batch to reach 10

### Queue Management:
- **Target:** 10 pending tasks
- **Minimum:** 5 pending tasks
- **Auto-create:** When queue drops below 5, creates enough to reach 10
- **Max per client:** 5 concurrent tasks

## Testing Instructions

### 1. Restart the Server
```bash
cd /Users/lalith/Documents/GitHub/oceans-four-driftcast/cs
# If using Docker
docker-compose restart api

# Or if running directly
cd server
source venv/bin/activate
uvicorn app.api.main:app --reload
```

### 2. Monitor Server Logs
Look for these log messages:
- `"Client {id} has X/5 active tasks"`
- `"Task {id} completed by client {id}. Active tasks now: X"`
- `"Queue below minimum. Creating X new tasks"`
- `"✓ Auto-created batch with X tasks"`

### 3. Check Queue Status
```bash
# Check current status
curl -s http://localhost:8000/api/v1/tasks/queue/status | jq

# Check active tasks detail
curl -s http://localhost:8000/api/v1/admin/debug/active-tasks | jq
```

### 4. Run Client
Start the client and watch it continuously process tasks. You should see:
- Tasks being completed quickly (100-200ms with fallback)
- New tasks being assigned immediately
- Queue size staying between 5-10
- No "task limit reached" messages

### 5. Verify Continuous Processing
```bash
# In one terminal, watch queue status
watch -n 2 'curl -s http://localhost:8000/api/v1/tasks/queue/status | jq'

# In another terminal, watch active tasks
watch -n 2 'curl -s http://localhost:8000/api/v1/admin/debug/active-tasks | jq ".total_active_tasks"'
```

## Expected Behavior

### Normal Operation:
```
Pending: 10 → 9 → 8 → 7 → 6 → 5 (triggers auto-create) → 10 → 9 → ...
Active:  0  → 1 → 2 → 1 → 2 → 1 → 2 → 1 → ...
```

### What You Should See:
1. **Client starts:** Requests tasks, gets assigned
2. **Tasks complete:** Active count decreases, pending decreases
3. **Queue reaches 5:** Server auto-creates batch, pending back to 10
4. **Client keeps working:** Continuous task flow, no interruptions

## Troubleshooting

### If Client Still Can't Get Tasks:

**Check active tasks:**
```bash
curl -s http://localhost:8000/api/v1/admin/debug/active-tasks | jq
```

Look for tasks with high `age_seconds` - these are stuck tasks.

**Clear stuck tasks:**
```bash
curl -X POST http://localhost:8000/api/v1/admin/tasks/clear-stuck
```

This will return all stuck tasks to the queue.

### If Queue Doesn't Auto-Fill:

Check the logs for:
- `"Queue below minimum"` messages
- Any errors in `_check_and_create_tasks()`
- Make sure at least one client is connected

### If WebSocket Messages Aren't Received:

Check:
1. Client is using WebSocket for task completion (not HTTP fallback)
2. WebSocket connection is established (client logs show "WebSocket connected")
3. Server logs show "Received task_completed" messages

## Configuration

Current settings in `task_manager.py`:
```python
target_pending_tasks = 10    # Keep queue at this level
min_pending_tasks = 5        # Trigger new batch when below this
max_tasks_per_client = 5     # Max concurrent tasks per client
particles_per_task = 10      # Particles in each task
```

Adjust these if needed for your testing requirements.

## Key Files Modified

1. `/cs/server/app/scheduler/task_manager.py` - Task scheduling logic
2. `/cs/server/app/api/main.py` - WebSocket message handling
3. `/cs/server/app/api/routes/admin.py` - Debug endpoint

## Next Steps

1. **Test** the continuous task processing
2. **Monitor** the logs to ensure tasks are completing properly
3. **Verify** queue auto-replenishment is working
4. **Check** that no tasks get stuck in active_tasks

If you still see issues, use the debug endpoint to inspect the active tasks and identify which tasks are not completing properly.

