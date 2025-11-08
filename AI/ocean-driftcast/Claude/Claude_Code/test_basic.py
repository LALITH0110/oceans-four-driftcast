#!/usr/bin/env python3
"""Quick test to verify make_all.py structure is correct."""

import numpy as np
from pathlib import Path

# Import components from make_all
import sys
sys.path.insert(0, str(Path(__file__).parent))

print("Testing basic components...")

# Test imports
from make_all import (
    NorthAtlanticGyreModel,
    Geography,
    create_particle_sources,
    ParticleSimulator,
    RANDOM_SEED
)

print("[OK] Imports successful")

# Test ocean model
ocean_model = NorthAtlanticGyreModel()
lons, lats, u, v = ocean_model.get_current_grid(resolution=5.0)
print(f"[OK] Ocean model initialized: grid shape {u.shape}")

# Test geography
geography = Geography()
print(f"[OK] Geography initialized: {len(geography.land_polygons)} land polygons")

# Test particle creation
rng = np.random.RandomState(RANDOM_SEED)
particles = create_particle_sources(rng)
print(f"[OK] Particles created: {len(particles['lon'])} particles")

# Test simulator initialization
simulator = ParticleSimulator(ocean_model, geography, rng)
print("[OK] Simulator initialized")

# Test velocity field at a point
u_test, v_test = ocean_model.velocity_field(
    np.array([-75.0]), np.array([35.0])
)
print(f"[OK] Velocity field test: u={u_test[0]:.3f}, v={v_test[0]:.3f} deg/day")

# Test one advection step
lon_test = np.array([-75.0])
lat_test = np.array([35.0])
lon_new, lat_new = simulator.advect_rk4(lon_test, lat_test, 1.0)
print(f"[OK] RK4 advection test: ({lon_test[0]:.2f}, {lat_test[0]:.2f}) -> ({lon_new[0]:.2f}, {lat_new[0]:.2f})")

# Test beaching detection
near_coast = geography.is_near_coast(-75.0, 35.0)
print(f"[OK] Beaching detection: point at (-75, 35) near coast = {near_coast}")

print("\n" + "="*50)
print("All basic tests passed!")
print("="*50)
print("\nThe make_all.py script structure is correct.")
print("To generate full outputs, run: python make_all.py")
print("(Expected runtime: 10-30 minutes)")
