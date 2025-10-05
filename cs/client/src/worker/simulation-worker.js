/**
 * Worker thread for processing ocean simulation tasks
 */
const { parentPort, workerData } = require('worker_threads');
const crypto = require('crypto');

class SimulationWorker {
    constructor(taskData) {
        this.taskData = taskData;
        this.startTime = Date.now();
    }
    
    async processTask() {
        try {
            console.log(`Processing task ${this.taskData.task_id} with ${this.taskData.particle_count} particles`);
            
            // Decode input data
            const inputData = Buffer.from(this.taskData.input_data, 'hex');
            
            // Parse simulation parameters
            const parameters = this.taskData.parameters;
            const particleCount = this.taskData.particle_count;
            
            // Simulate ocean plastic drift
            const results = await this.simulateParticleDrift(inputData, parameters, particleCount);
            
            // Encode results
            const resultData = this.encodeResults(results);
            
            console.log(`Task ${this.taskData.task_id} completed successfully`);
            
            return {
                success: true,
                result_data: resultData,
                particle_count: particleCount,
                execution_time: Date.now() - this.startTime,
                metadata: {
                    particles_processed: results.length,
                    simulation_steps: parameters.time_steps || 100
                }
            };
            
        } catch (error) {
            console.error(`Task ${this.taskData.task_id} failed:`, error);
            
            return {
                success: false,
                error: error.message,
                execution_time: Date.now() - this.startTime
            };
        }
    }
    
    async simulateParticleDrift(inputData, parameters, particleCount) {
        // This is a simplified simulation for demonstration
        // In a real implementation, this would use actual ocean current data
        // and physics-based particle tracking algorithms
        
        const results = [];
        const timeSteps = parameters.time_steps || 100;
        const timeStep = parameters.time_step || 3600; // 1 hour in seconds
        
        // Initialize particles with random starting positions
        const particles = this.initializeParticles(particleCount, parameters);
        
        // Simulate particle movement over time
        for (let step = 0; step < timeSteps; step++) {
            await this.simulateTimeStep(particles, step, timeStep, parameters);
            
            // Report progress occasionally
            if (step % 10 === 0) {
                const progress = (step / timeSteps) * 100;
                console.log(`Task ${this.taskData.task_id} progress: ${progress.toFixed(1)}%`);
            }
            
            // Add some realistic processing delay
            if (step % 20 === 0) {
                await this.sleep(10); // 10ms delay every 20 steps
            }
        }
        
        // Record final positions
        for (const particle of particles) {
            results.push({
                id: particle.id,
                initial_position: particle.initialPosition,
                final_position: particle.position,
                trajectory: particle.trajectory,
                beached: particle.beached,
                distance_traveled: particle.distanceTraveled
            });
        }
        
        return results;
    }
    
    initializeParticles(count, parameters) {
        const particles = [];
        const bounds = parameters.spatial_bounds || {
            minLat: -90, maxLat: 90,
            minLon: -180, maxLon: 180
        };
        
        for (let i = 0; i < count; i++) {
            const lat = bounds.minLat + Math.random() * (bounds.maxLat - bounds.minLat);
            const lon = bounds.minLon + Math.random() * (bounds.maxLon - bounds.minLon);
            
            particles.push({
                id: i,
                position: { lat, lon },
                initialPosition: { lat, lon },
                velocity: { u: 0, v: 0 }, // East-West, North-South velocity
                trajectory: [{ lat, lon, time: 0 }],
                beached: false,
                distanceTraveled: 0
            });
        }
        
        return particles;
    }
    
    async simulateTimeStep(particles, step, timeStep, parameters) {
        for (const particle of particles) {
            if (particle.beached) {
                continue; // Skip beached particles
            }
            
            // Simulate ocean currents (simplified)
            const currentVelocity = this.getOceanCurrent(particle.position, step);
            
            // Add wind effects (simplified)
            const windEffect = this.getWindEffect(particle.position, step);
            
            // Add random diffusion
            const diffusion = this.getRandomDiffusion();
            
            // Update particle velocity
            particle.velocity.u = currentVelocity.u + windEffect.u + diffusion.u;
            particle.velocity.v = currentVelocity.v + windEffect.v + diffusion.v;
            
            // Update particle position
            const oldPosition = { ...particle.position };
            particle.position.lat += particle.velocity.v * timeStep / 111000; // Rough conversion
            particle.position.lon += particle.velocity.u * timeStep / (111000 * Math.cos(particle.position.lat * Math.PI / 180));
            
            // Calculate distance traveled
            const distance = this.calculateDistance(oldPosition, particle.position);
            particle.distanceTraveled += distance;
            
            // Check for beaching (simplified)
            if (this.isNearCoast(particle.position)) {
                particle.beached = Math.random() < 0.1; // 10% chance of beaching near coast
            }
            
            // Record trajectory point
            particle.trajectory.push({
                lat: particle.position.lat,
                lon: particle.position.lon,
                time: step * timeStep
            });
        }
    }
    
    getOceanCurrent(position, timeStep) {
        // Simplified ocean current model
        // In reality, this would use actual oceanographic data
        const lat = position.lat * Math.PI / 180;
        const lon = position.lon * Math.PI / 180;
        const time = timeStep * 0.01;
        
        return {
            u: 0.1 * Math.sin(lat * 2) * Math.cos(lon + time), // East-West current
            v: 0.05 * Math.cos(lat) * Math.sin(lon * 2 + time)  // North-South current
        };
    }
    
    getWindEffect(position, timeStep) {
        // Simplified wind effect (about 3% of wind speed affects surface particles)
        const windFactor = 0.03;
        const lat = position.lat * Math.PI / 180;
        const time = timeStep * 0.005;
        
        return {
            u: windFactor * 0.2 * Math.sin(lat + time),
            v: windFactor * 0.1 * Math.cos(lat * 2 + time)
        };
    }
    
    getRandomDiffusion() {
        // Random diffusion to simulate turbulence
        const diffusionStrength = 0.01;
        
        return {
            u: diffusionStrength * (Math.random() - 0.5),
            v: diffusionStrength * (Math.random() - 0.5)
        };
    }
    
    isNearCoast(position) {
        // Simplified coast detection
        // In reality, this would use actual coastline data
        const lat = Math.abs(position.lat);
        const lon = Math.abs(position.lon);
        
        // Assume some areas are near coasts
        return (lat > 60) || // Near poles
               (lat < 10 && lon > 170) || // Pacific islands
               (lat > 30 && lat < 50 && lon > 120 && lon < 140); // Japan area
    }
    
    calculateDistance(pos1, pos2) {
        // Haversine formula for distance calculation
        const R = 6371000; // Earth's radius in meters
        const lat1 = pos1.lat * Math.PI / 180;
        const lat2 = pos2.lat * Math.PI / 180;
        const deltaLat = (pos2.lat - pos1.lat) * Math.PI / 180;
        const deltaLon = (pos2.lon - pos1.lon) * Math.PI / 180;
        
        const a = Math.sin(deltaLat / 2) * Math.sin(deltaLat / 2) +
                  Math.cos(lat1) * Math.cos(lat2) *
                  Math.sin(deltaLon / 2) * Math.sin(deltaLon / 2);
        const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
        
        return R * c;
    }
    
    encodeResults(results) {
        // Encode results as hex string for transmission
        const jsonString = JSON.stringify(results);
        return Buffer.from(jsonString).toString('hex');
    }
    
    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// Main worker execution
async function main() {
    if (!workerData) {
        throw new Error('No task data provided to worker');
    }
    
    const worker = new SimulationWorker(workerData);
    const result = await worker.processTask();
    
    // Send result back to main thread
    parentPort.postMessage(result);
}

// Handle uncaught errors
process.on('uncaughtException', (error) => {
    console.error('Uncaught exception in worker:', error);
    parentPort.postMessage({
        success: false,
        error: error.message,
        execution_time: Date.now() - (workerData?.startTime || Date.now())
    });
});

// Start processing
main().catch((error) => {
    console.error('Worker main function failed:', error);
    parentPort.postMessage({
        success: false,
        error: error.message,
        execution_time: 0
    });
});
