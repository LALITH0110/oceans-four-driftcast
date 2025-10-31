/**
 * Server communication for API calls and WebSocket connection
 */
const axios = require('axios');
const https = require('https');
const WebSocket = require('ws');
const log = require('electron-log');
const Store = require('electron-store');

const store = new Store();

class ServerCommunicator {
    constructor() {
        // Production server
        this.serverUrl = 'https://system76.rice.iit.edu';
        // Local development (uncomment for local testing)
        // this.serverUrl = 'http://localhost:8000';
        
        this.wsUrl = this.serverUrl.replace('http', 'ws').replace('https', 'wss');
        this.clientId = null;
        this.token = null;
        this.websocket = null;
        this.isConnected = false;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectInterval = 5000; // 5 seconds
        
        // Setup axios defaults
        this.api = axios.create({
            baseURL: this.serverUrl,
            timeout: 30000,
            headers: {
                'Content-Type': 'application/json'
            },
            // Allow self-signed certificates for development/production with self-signed certs
            httpsAgent: new https.Agent({
                rejectUnauthorized: false
            })
        });
        
        // Add request interceptor for authentication
        this.api.interceptors.request.use((config) => {
            if (this.token) {
                config.headers.Authorization = `Bearer ${this.token}`;
            }
            return config;
        });
        
        // Add response interceptor for error handling
        this.api.interceptors.response.use(
            (response) => response,
            (error) => {
                log.error('API request failed:', error.message);
                if (error.response?.status === 401) {
                    // Token expired, clear credentials
                    this.clearCredentials();
                }
                throw error;
            }
        );
    }
    
    setCredentials(clientId, token) {
        this.clientId = clientId;
        this.token = token;
        log.info(`Credentials set for client ${clientId}`);
    }
    
    clearCredentials() {
        this.clientId = null;
        this.token = null;
        store.delete('clientId');
        store.delete('token');
        log.info('Credentials cleared');
    }
    
    async registerClient(clientData) {
        try {
            log.info('Registering client with server...');
            
            // Clean and validate data before sending
            const registrationData = {
                name: clientData.name || 'Unknown Client',
                public_key: clientData.publicKey || 'default-key',
                capabilities: clientData.capabilities || {},
                system_info: clientData.systemInfo || {}
            };
            
            log.info('Registration data:', JSON.stringify(registrationData, null, 2));
            
            const response = await this.api.post('/api/v1/clients/register', registrationData);
            
            const { client_id, token, expires_in } = response.data;
            
            this.setCredentials(client_id, token);
            
            // Connect WebSocket
            await this.connectWebSocket();
            
            log.info('Client registered successfully');
            
            return {
                success: true,
                clientId: client_id,
                token: token,
                expiresIn: expires_in
            };
            
        } catch (error) {
            log.error('Registration failed:', error.message);
            return {
                success: false,
                error: error.response?.data?.detail || error.message
            };
        }
    }
    
    async verifyToken() {
        if (!this.token) {
            return false;
        }
        
        try {
            await this.api.get('/api/v1/clients/me');
            return true;
        } catch (error) {
            log.warn('Token verification failed:', error.message);
            return false;
        }
    }
    
    async connectWebSocket() {
        if (!this.clientId) {
            throw new Error('No client ID available for WebSocket connection');
        }
        
        return new Promise((resolve, reject) => {
            try {
                const wsUrl = `${this.wsUrl}/ws/client/${this.clientId}`;
                // Allow self-signed certificates
                this.websocket = new WebSocket(wsUrl, {
                    rejectUnauthorized: false
                });
                
                this.websocket.on('open', () => {
                    log.info('WebSocket connected');
                    this.isConnected = true;
                    this.reconnectAttempts = 0;
                    
                    // Start heartbeat
                    this.startHeartbeat();
                    
                    resolve();
                });
                
                this.websocket.on('message', async (data) => {
                    try {
                        const message = JSON.parse(data);
                        await this.handleWebSocketMessage(message);
                    } catch (error) {
                        log.error('Error parsing WebSocket message:', error);
                    }
                });
                
                this.websocket.on('close', () => {
                    log.warn('WebSocket disconnected');
                    this.isConnected = false;
                    this.stopHeartbeat();
                    
                    // Attempt reconnection
                    this.scheduleReconnect();
                });
                
                this.websocket.on('error', (error) => {
                    log.error('WebSocket error:', error);
                    reject(error);
                });
                
            } catch (error) {
                reject(error);
            }
        });
    }
    
    async handleWebSocketMessage(message) {
        log.debug('Received WebSocket message:', message.type);
        
        switch (message.type) {
            case 'heartbeat_ack':
                // Heartbeat acknowledged
                break;
                
            case 'task_assignment':
                // New task assigned
                log.info(`Received task assignment: ${message.task.task_id}`);
                await this.handleTaskAssignment(message.task);
                break;
                
            case 'no_tasks_available':
                log.debug('No tasks available from server');
                break;
                
            case 'task_completion_ack':
                log.info(`Task completion acknowledged: ${message.task_id}`);
                break;
                
            case 'task_failure_ack':
                log.info(`Task failure acknowledged: ${message.task_id}`);
                break;
                
            case 'system_announcement':
                log.info(`System announcement: ${message.message}`);
                break;
                
            case 'ping':
                // Respond to ping
                this.sendWebSocketMessage({
                    type: 'pong',
                    timestamp: message.timestamp
                });
                break;
                
            default:
                log.warn('Unknown WebSocket message type:', message.type);
        }
    }
    
    async handleTaskAssignment(taskData) {
        log.info(`Processing task assignment: ${taskData.task_id}`);
        
        // Emit event for task manager to handle
        const { ipcMain } = require('electron');
        ipcMain.emit('task-assignment', taskData);
    }
    
    sendWebSocketMessage(message) {
        if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
            log.debug(`Sending WebSocket message: ${message.type}`);
            this.websocket.send(JSON.stringify(message));
        } else {
            log.warn(`WebSocket not connected, cannot send message. ReadyState: ${this.websocket?.readyState}`);
        }
    }
    
    startHeartbeat() {
        this.heartbeatInterval = setInterval(() => {
            this.sendWebSocketMessage({
                type: 'heartbeat',
                timestamp: new Date().toISOString()
            });
        }, 30000); // Every 30 seconds
    }
    
    stopHeartbeat() {
        if (this.heartbeatInterval) {
            clearInterval(this.heartbeatInterval);
            this.heartbeatInterval = null;
        }
    }
    
    scheduleReconnect() {
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            log.error('Max reconnection attempts reached');
            return;
        }
        
        this.reconnectAttempts++;
        const delay = this.reconnectInterval * this.reconnectAttempts;
        
        log.info(`Scheduling reconnection attempt ${this.reconnectAttempts} in ${delay}ms`);
        
        setTimeout(async () => {
            try {
                await this.connectWebSocket();
            } catch (error) {
                log.error('Reconnection failed:', error.message);
                this.scheduleReconnect();
            }
        }, delay);
    }
    
    async requestTask() {
        try {
            // Temporarily force HTTP API for debugging
            // TODO: Remove this and restore WebSocket after fixing the issue
            if (false && this.isConnected && this.websocket && this.websocket.readyState === 1) {
                log.debug(`Sending WebSocket task_request (readyState: ${this.websocket.readyState})`);
                this.sendWebSocketMessage({
                    type: 'task_request'
                });
                return null; // Task will come via WebSocket message
            } else {
                log.info(`Using HTTP API instead of WebSocket for task requests`);
            }
            
            // Fallback to HTTP API
            log.info('Using HTTP API fallback for task request');
            const response = await this.api.get('/api/v1/tasks/available');
            return response.data;
            
        } catch (error) {
            if (error.response?.status !== 404) {
                log.error('Error requesting task:', error.message);
            }
            return null;
        }
    }
    
    async submitTaskResult(taskId, resultData, executionTime) {
        try {
            // Convert result data to hex string for server
            const resultJson = JSON.stringify(resultData);
            const resultHex = Buffer.from(resultJson, 'utf8').toString('hex');
            
            // Try WebSocket first
            if (this.isConnected) {
                this.sendWebSocketMessage({
                    type: 'task_completed',
                    task_id: taskId,
                    result_data: resultHex,
                    execution_time: executionTime / 1000 // Convert to seconds
                });
                return;
            }
            
            // Fallback to HTTP API
            await this.api.post('/api/v1/tasks/complete', {
                task_id: taskId,
                result_data: resultHex,
                execution_time: executionTime / 1000,
                metadata: {}
            });
            
            log.info(`Task result submitted: ${taskId}`);
            
        } catch (error) {
            log.error(`Error submitting task result for ${taskId}:`, error.message);
            throw error;
        }
    }
    
    async reportTaskFailure(taskId, errorMessage) {
        try {
            // Try WebSocket first
            if (this.isConnected) {
                this.sendWebSocketMessage({
                    type: 'task_failed',
                    task_id: taskId,
                    error: errorMessage
                });
                return;
            }
            
            // Fallback to HTTP API
            await this.api.post('/api/v1/tasks/fail', {
                task_id: taskId,
                error_message: errorMessage,
                metadata: {}
            });
            
            log.info(`Task failure reported: ${taskId}`);
            
        } catch (error) {
            log.error(`Error reporting task failure for ${taskId}:`, error.message);
            throw error;
        }
    }

    async clearStuckTasks() {
        try {
            const response = await this.api.post('/api/v1/admin/tasks/clear-stuck');
            return response.data;
        } catch (error) {
            log.error('Error calling clear-stuck:', error.message);
            return null;
        }
    }
    
    async sendHeartbeat() {
        try {
            await this.api.put('/api/v1/clients/heartbeat');
        } catch (error) {
            log.error('Heartbeat failed:', error.message);
        }
    }
    
    async disconnect() {
        log.info('Disconnecting from server...');
        
        this.stopHeartbeat();
        
        if (this.websocket) {
            this.websocket.close();
            this.websocket = null;
        }
        
        this.isConnected = false;
        log.info('Disconnected from server');
    }
    
    getConnectionStatus() {
        return {
            isConnected: this.isConnected,
            serverUrl: this.serverUrl,
            clientId: this.clientId,
            hasToken: !!this.token,
            reconnectAttempts: this.reconnectAttempts
        };
    }
    
    updateServerUrl(newUrl) {
        this.serverUrl = newUrl;
        this.wsUrl = newUrl.replace('http', 'ws');
        this.api.defaults.baseURL = newUrl;
        store.set('serverUrl', newUrl);
        log.info(`Server URL updated to: ${newUrl}`);
    }
}

module.exports = ServerCommunicator;
