/**
 * Task management for processing ocean simulation tasks
 */
const { Worker } = require('worker_threads');
const path = require('path');
const log = require('electron-log');
const cron = require('node-cron');

class TaskManager {
    constructor(serverCommunicator) {
        this.serverCommunicator = serverCommunicator;
        this.isRunning = false;
        this.isPaused = false;
        this.activeTasks = new Map();
        this.statistics = {
            completed: 0,
            failed: 0,
            active: 0,
            totalExecutionTime: 0,
            successRate: 0
        };

        this.maxConcurrentTasks = 2; // Configurable
        this.baseConcurrentTasks = 2; // Base value
        this.turboConcurrentTasks = 4; // Turbo mode value
        this.isTurboMode = false;
        this.taskTimeout = 30 * 60 * 1000; // 30 minutes
        
        // Cron job for periodic task requests
        this.taskRequestJob = null;
        // Cron job for periodic stuck-task clearing
        this.clearStuckJob = null;
        
        // Prevent duplicate task requests
        this.isRequestingTask = false;
        this.lastRequestTime = 0;
    }
    
    async start() {
        if (this.isRunning) {
            log.warn('Task manager already running');
            return;
        }
        
        log.info('Starting task manager...');
        this.isRunning = true;
        this.isPaused = false;
        
        // Listen for task assignments from WebSocket
        const { ipcMain } = require('electron');
        ipcMain.on('task-assignment', (taskData) => {
            log.info(`Task manager received assignment: ${taskData.task_id}`);
            this.processTask(taskData);
        });
        
        // Start periodic task requests (every 10 seconds as backup)
        this.taskRequestJob = cron.schedule('*/10 * * * * *', async () => {
            if (!this.isPaused && this.activeTasks.size < this.maxConcurrentTasks) {
                log.debug('Periodic task request check...');
                await this.requestTask();
            }
        }, {
            scheduled: false
        });
        
        this.taskRequestJob.start();
        
        // Start periodic clear-stuck (every 10 seconds)
        this.clearStuckJob = cron.schedule('*/10 * * * * *', async () => {
            try {
                const result = await this.serverCommunicator.clearStuckTasks();
                if (result?.cleared_count > 0) {
                    log.warn(`Auto clear-stuck requeued ${result.cleared_count} tasks`);
                }
            } catch (e) {
                log.debug('clear-stuck call failed (ignored):', e.message);
            }
        }, { scheduled: true });
        
        // Request initial task immediately
        setTimeout(() => {
            if (!this.isPaused && this.activeTasks.size < this.maxConcurrentTasks) {
                log.info('Requesting initial task...');
                this.requestTask();
            }
        }, 1000); // 1 second delay to ensure WebSocket is connected
        
        log.info('Task manager started');
    }
    
    async stop() {
        if (!this.isRunning) {
            return;
        }
        
        log.info('Stopping task manager...');
        this.isRunning = false;
        
        // Stop cron job
        if (this.taskRequestJob) {
            this.taskRequestJob.stop();
        }
        if (this.clearStuckJob) {
            this.clearStuckJob.stop();
        }
        
        // Wait for active tasks to complete or timeout
        const activeTaskIds = Array.from(this.activeTasks.keys());
        for (const taskId of activeTaskIds) {
            await this.cancelTask(taskId);
        }
        
        log.info('Task manager stopped');
    }
    
    async pause() {
        this.isPaused = true;
        log.info('Task processing paused');

        // Cancel all active tasks to free up CPU
        const activeTaskIds = Array.from(this.activeTasks.keys());
        for (const taskId of activeTaskIds) {
            await this.cancelTask(taskId, 'paused');
        }

        log.info(`Cancelled ${activeTaskIds.length} active tasks`);
    }

    resume() {
        this.isPaused = false;
        log.info('Task processing resumed');

        // Immediately request new tasks after resuming
        if (this.activeTasks.size < this.maxConcurrentTasks) {
            setTimeout(() => this.requestTask(), 100);
        }
    }
    
    async requestTask() {
        if (!this.isRunning || this.isPaused) {
            return null;
        }
        
        if (this.activeTasks.size >= this.maxConcurrentTasks) {
            log.debug('Max concurrent tasks reached, skipping request');
            return null;
        }
        
        // Prevent duplicate requests within 1 second
        const now = Date.now();
        if (this.isRequestingTask || (now - this.lastRequestTime) < 1000) {
            log.debug('Task request already in progress or too recent, skipping');
            return null;
        }
        
        this.isRequestingTask = true;
        this.lastRequestTime = now;
        
        try {
            log.info('Requesting task from server...');
            // Request task from server
            const taskData = await this.serverCommunicator.requestTask();
            
            if (taskData) {
                await this.processTask(taskData);
                return taskData;
            }
            
        } catch (error) {
            log.error('Error requesting task:', error);
        } finally {
            this.isRequestingTask = false;
        }
        
        return null;
    }
    
    async processTask(taskData) {
        const taskId = taskData.task_id;
        const startTime = Date.now();
        
        log.info(`Starting task ${taskId}`);
        
        try {
            // Create worker thread for task processing
            const worker = new Worker(
                path.join(__dirname, '../worker/simulation-worker.js'),
                {
                    workerData: taskData
                }
            );
            
            // Store task info
            this.activeTasks.set(taskId, {
                worker,
                startTime,
                taskData
            });
            
            this.updateStatistics();
            
            // Set up timeout
            const timeout = setTimeout(() => {
                this.cancelTask(taskId, 'timeout');
            }, this.taskTimeout);
            
            // Handle worker completion
            worker.on('message', async (result) => {
                clearTimeout(timeout);
                await this.handleTaskCompletion(taskId, result, startTime);
            });
            
            // Handle worker error
            worker.on('error', async (error) => {
                clearTimeout(timeout);
                await this.handleTaskFailure(taskId, error, startTime);
            });
            
            // Handle worker exit
            worker.on('exit', (code) => {
                if (code !== 0) {
                    log.error(`Worker for task ${taskId} exited with code ${code}`);
                }
            });
            
        } catch (error) {
            log.error(`Error processing task ${taskId}:`, error);
            await this.handleTaskFailure(taskId, error, startTime);
        }
    }
    
    async handleTaskCompletion(taskId, result, startTime) {
        const executionTime = Date.now() - startTime;
        
        log.info(`Task ${taskId} completed in ${executionTime}ms`);
        
        try {
            // Send results to server
            await this.serverCommunicator.submitTaskResult(taskId, result, executionTime);
            
            // Update statistics
            this.statistics.completed++;
            this.statistics.totalExecutionTime += executionTime;
            this.updateSuccessRate();
            
        } catch (error) {
            log.error(`Error submitting results for task ${taskId}:`, error);
            this.statistics.failed++;
        } finally {
            // Clean up
            this.activeTasks.delete(taskId);
            this.updateStatistics();
            
            // Immediately request a new task if we have capacity
            if (!this.isPaused && this.activeTasks.size < this.maxConcurrentTasks) {
                log.info('Task completed, immediately requesting new task...');
                setTimeout(() => this.requestTask(), 100); // Small delay to avoid race conditions
            }
        }
    }
    
    async handleTaskFailure(taskId, error, startTime) {
        const executionTime = Date.now() - startTime;
        
        log.error(`Task ${taskId} failed after ${executionTime}ms:`, error);
        
        try {
            // Report failure to server
            await this.serverCommunicator.reportTaskFailure(taskId, error.message);
            
        } catch (reportError) {
            log.error(`Error reporting task failure for ${taskId}:`, reportError);
        } finally {
            // Update statistics
            this.statistics.failed++;
            this.updateSuccessRate();
            
            // Clean up
            this.activeTasks.delete(taskId);
            this.updateStatistics();
            
            // Immediately request a new task if we have capacity
            if (!this.isPaused && this.activeTasks.size < this.maxConcurrentTasks) {
                log.info('Task failed, immediately requesting new task...');
                setTimeout(() => this.requestTask(), 100); // Small delay to avoid race conditions
            }
        }
    }
    
    async cancelTask(taskId, reason = 'cancelled') {
        const taskInfo = this.activeTasks.get(taskId);
        if (!taskInfo) {
            return;
        }
        
        log.info(`Cancelling task ${taskId}: ${reason}`);
        
        try {
            // Terminate worker
            await taskInfo.worker.terminate();
            
            // Report cancellation to server
            await this.serverCommunicator.reportTaskFailure(taskId, `Task cancelled: ${reason}`);
            
        } catch (error) {
            log.error(`Error cancelling task ${taskId}:`, error);
        } finally {
            this.activeTasks.delete(taskId);
            this.updateStatistics();
        }
    }
    
    updateStatistics() {
        this.statistics.active = this.activeTasks.size;
    }
    
    updateSuccessRate() {
        const total = this.statistics.completed + this.statistics.failed;
        this.statistics.successRate = total > 0 ? 
            (this.statistics.completed / total) * 100 : 0;
    }
    
    getStatistics() {
        return {
            ...this.statistics,
            averageExecutionTime: this.statistics.completed > 0 ? 
                this.statistics.totalExecutionTime / this.statistics.completed : 0
        };
    }
    
    setMaxConcurrentTasks(max) {
        this.maxConcurrentTasks = Math.max(1, Math.min(max, 10)); // Limit between 1-10
        log.info(`Max concurrent tasks set to ${this.maxConcurrentTasks}`);
    }

    setTurboMode(enabled) {
        this.isTurboMode = enabled;
        this.maxConcurrentTasks = enabled ? this.turboConcurrentTasks : this.baseConcurrentTasks;
        log.info(`Turbo mode ${enabled ? 'enabled' : 'disabled'} - concurrent tasks: ${this.maxConcurrentTasks}`);

        // If we enabled turbo and have capacity, request more tasks immediately
        if (enabled && !this.isPaused && this.activeTasks.size < this.maxConcurrentTasks) {
            const tasksToRequest = this.maxConcurrentTasks - this.activeTasks.size;
            for (let i = 0; i < tasksToRequest; i++) {
                setTimeout(() => this.requestTask(), i * 200);
            }
        }
    }
    
    getActiveTasks() {
        return Array.from(this.activeTasks.entries()).map(([taskId, info]) => ({
            taskId,
            startTime: info.startTime,
            duration: Date.now() - info.startTime,
            particleCount: info.taskData.particle_count
        }));
    }
}

module.exports = TaskManager;
