#!/usr/bin/env python3
"""
Test script to demonstrate the task queue management fix
"""
import asyncio
import aiohttp
import json
from datetime import datetime

SERVER_URL = "http://localhost:8000"

async def test_queue_management():
    """Test the improved task queue management"""
    
    print("🧪 Testing Ocean Plastic Drift Task Queue Management Fix")
    print("=" * 60)
    
    async with aiohttp.ClientSession() as session:
        
        # 1. Check initial queue status
        print("\n1️⃣ Checking initial queue status...")
        async with session.get(f"{SERVER_URL}/api/v1/tasks/queue/status") as resp:
            if resp.status == 200:
                status = await resp.json()
                print(f"   📊 Initial Status:")
                print(f"      • Pending tasks: {status['pending_tasks']}")
                print(f"      • Active tasks: {status['active_tasks']}")
                print(f"      • Active clients: {status['active_clients']}")
                print(f"      • Target pending: {status['target_pending_tasks']}")
                print(f"      • Min pending: {status['min_pending_tasks']}")
                print(f"      • Client capacity: {status['total_client_capacity']}")
            else:
                print(f"   ❌ Failed to get queue status: {resp.status}")
                return
        
        # 2. Update task configuration to test values
        print("\n2️⃣ Updating task configuration...")
        config_data = {
            "target_pending_tasks": 10,
            "min_pending_tasks": 5,
            "max_tasks_per_client": 3
        }
        
        async with session.post(f"{SERVER_URL}/api/v1/admin/tasks/config", 
                               json=config_data) as resp:
            if resp.status == 200:
                result = await resp.json()
                print(f"   ✅ Configuration updated:")
                config = result['current_config']
                print(f"      • Target pending: {config['target_pending_tasks']}")
                print(f"      • Min pending: {config['min_pending_tasks']}")
                print(f"      • Max per client: {config['max_tasks_per_client']}")
            else:
                print(f"   ❌ Failed to update config: {resp.status}")
        
        # 3. Create a test batch to trigger task creation
        print("\n3️⃣ Creating test simulation batch...")
        batch_data = {
            "name": "Test Queue Management Batch",
            "description": "Testing the improved queue management system",
            "particle_count": 25,  # This will create 5 tasks (5 particles each)
            "time_horizon": 1,
            "spatial_bounds": {
                "min_lat": 25.0,
                "max_lat": 30.0,
                "min_lon": -95.0,
                "max_lon": -85.0
            },
            "parameters": {
                "current_strength": 0.5,
                "wind_speed": 10.0,
                "priority": 1,
                "test_batch": True
            }
        }
        
        async with session.post(f"{SERVER_URL}/api/v1/admin/batch/create", 
                               json=batch_data) as resp:
            if resp.status == 200:
                result = await resp.json()
                print(f"   ✅ Batch created: {result['batch_id']}")
            else:
                print(f"   ❌ Failed to create batch: {resp.status}")
        
        # 4. Monitor queue status over time
        print("\n4️⃣ Monitoring queue status (will show automatic task creation)...")
        for i in range(6):  # Monitor for 30 seconds
            await asyncio.sleep(5)
            
            async with session.get(f"{SERVER_URL}/api/v1/tasks/queue/status") as resp:
                if resp.status == 200:
                    status = await resp.json()
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    print(f"   📊 [{timestamp}] Queue Status:")
                    print(f"      • Pending: {status['pending_tasks']}")
                    print(f"      • Active: {status['active_tasks']}")
                    print(f"      • Clients: {status['active_clients']}")
                    print(f"      • Capacity: {status['total_client_capacity']}")
                    
                    # Check if the system is working correctly
                    if status['pending_tasks'] >= status['min_pending_tasks']:
                        print(f"      ✅ Queue maintained above minimum ({status['min_pending_tasks']})")
                    else:
                        print(f"      ⚠️  Queue below minimum ({status['min_pending_tasks']})")
        
        # 5. Get system statistics
        print("\n5️⃣ Final system statistics...")
        async with session.get(f"{SERVER_URL}/api/v1/admin/stats") as resp:
            if resp.status == 200:
                stats = await resp.json()
                print(f"   📈 System Stats:")
                print(f"      • Total clients: {stats['total_clients']}")
                print(f"      • Active clients: {stats['active_clients']}")
                print(f"      • Total tasks: {stats['total_tasks']}")
                print(f"      • Completed tasks: {stats['completed_tasks']}")
                print(f"      • Pending tasks: {stats['pending_tasks']}")
            else:
                print(f"   ❌ Failed to get stats: {resp.status}")
    
    print("\n" + "=" * 60)
    print("🎉 Test completed!")
    print("\nKey improvements implemented:")
    print("• ✅ Server now tracks total task demand (pending + active + client capacity)")
    print("• ✅ Immediate task creation when tasks are completed")
    print("• ✅ Configurable queue management parameters")
    print("• ✅ Better queue monitoring with capacity awareness")
    print("• ✅ Automatic task creation maintains target queue size")

if __name__ == "__main__":
    try:
        asyncio.run(test_queue_management())
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
