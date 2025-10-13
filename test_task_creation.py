#!/usr/bin/env python3
"""
Test script to create a task on the cloud server for the client to process
"""

import requests
import json
import time

# Server configuration
SERVER_URL = "http://system80.rice.iit.edu:8000"

def create_test_task():
    """Create a test drift simulation task"""
    
    # First, let's check if the server is accessible
    try:
        response = requests.get(f"{SERVER_URL}/")
        print(f"Server status: {response.json()}")
    except Exception as e:
        print(f"Error connecting to server: {e}")
        return
    
    # Create a test task via the admin endpoint
    task_data = {
        "name": "Test RL Drift Simulation",
        "particle_count": 100,
        "time_horizon": 48,  # 48 hours
        "spatial_bounds": {
            "min_lat": 30.0,
            "max_lat": 40.0,
            "min_lon": -80.0,
            "max_lon": -70.0
        },
        "parameters": {
            "time_steps": 100,
            "time_step": 3600,
            "priority": 1
        }
    }
    
    try:
        # Try to create a simulation batch
        print("Creating test simulation batch...")
        response = requests.post(
            f"{SERVER_URL}/api/v1/admin/batch/create",
            json=task_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Task created successfully!")
            print(f"Batch ID: {result.get('batch_id', 'Unknown')}")
        else:
            print(f"❌ Failed to create task: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"Error creating task: {e}")

def check_available_tasks():
    """Check what tasks are available"""
    try:
        response = requests.get(f"{SERVER_URL}/api/v1/tasks/available")
        if response.status_code == 200:
            tasks = response.json()
            print(f"Available tasks: {len(tasks) if isinstance(tasks, list) else 'Unknown'}")
            if tasks:
                print("Tasks:", json.dumps(tasks, indent=2))
        else:
            print(f"No tasks available or error: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Error checking tasks: {e}")

if __name__ == "__main__":
    print("🌊 Ocean Plastic Forecast - Task Creation Test")
    print("=" * 50)
    
    # Check server status
    print("\n1. Checking server status...")
    create_test_task()
    
    print("\n2. Checking available tasks...")
    check_available_tasks()
    
    print("\n✅ Test completed!")
    print("\nNow your client should be able to:")
    print("- Connect to the server")
    print("- Request and receive tasks") 
    print("- Run RL-enhanced simulations")
    print("- Send results back with your user ID")
