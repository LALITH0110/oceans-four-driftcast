/**
 * Renderer process for Ocean Plastic Forecast Client UI
 */
const { ipcRenderer } = require('electron');

class ClientUI {
    constructor() {
        this.isRegistered = false;
        this.isPaused = false;
        this.settings = {};
        
        this.initializeUI();
        this.setupEventListeners();
        this.loadSettings();
    }
    
    initializeUI() {
        // Get DOM elements
        this.elements = {
            registrationSection: document.getElementById('registrationSection'),
            dashboardSection: document.getElementById('dashboardSection'),
            registrationForm: document.getElementById('registrationForm'),
            settingsModal: document.getElementById('settingsModal'),
            settingsForm: document.getElementById('settingsForm'),
            
            // Status elements
            statusIndicator: document.getElementById('statusIndicator'),
            statusText: document.getElementById('statusText'),
            
            // System info
            cpuInfo: document.getElementById('cpuInfo'),
            memoryInfo: document.getElementById('memoryInfo'),
            platformInfo: document.getElementById('platformInfo'),
            
            // Task stats
            completedTasks: document.getElementById('completedTasks'),
            failedTasks: document.getElementById('failedTasks'),
            activeTasks: document.getElementById('activeTasks'),
            successRate: document.getElementById('successRate'),
            
            // Resource usage
            cpuUsageFill: document.getElementById('cpuUsageFill'),
            cpuUsageText: document.getElementById('cpuUsageText'),
            memoryUsageFill: document.getElementById('memoryUsageFill'),
            memoryUsageText: document.getElementById('memoryUsageText'),
            
            // Controls
            pauseBtn: document.getElementById('pauseBtn'),
            requestTaskBtn: document.getElementById('requestTaskBtn'),
            settingsBtn: document.getElementById('settingsBtn'),
            
            // Activity log
            activityLog: document.getElementById('activityLog'),
            
            // Settings
            maxCpuUsage: document.getElementById('maxCpuUsage'),
            maxCpuValue: document.getElementById('maxCpuValue'),
            maxMemoryUsage: document.getElementById('maxMemoryUsage'),
            autoStart: document.getElementById('autoStart'),
            runInBackground: document.getElementById('runInBackground')
        };
        
        this.logActivity('Application started', 'info');
    }
    
    setupEventListeners() {
        // Registration form
        this.elements.registrationForm.addEventListener('submit', (e) => {
            e.preventDefault();
            this.handleRegistration();
        });
        
        // Control buttons
        this.elements.pauseBtn.addEventListener('click', () => {
            this.toggleProcessing();
        });
        
        this.elements.requestTaskBtn.addEventListener('click', () => {
            this.requestTask();
        });
        
        this.elements.settingsBtn.addEventListener('click', () => {
            this.showSettings();
        });
        
        // Settings modal
        document.getElementById('closeSettings').addEventListener('click', () => {
            this.hideSettings();
        });
        
        document.getElementById('cancelSettings').addEventListener('click', () => {
            this.hideSettings();
        });
        
        document.getElementById('saveSettings').addEventListener('click', () => {
            this.saveSettings();
        });
        
        // Settings form inputs
        this.elements.maxCpuUsage.addEventListener('input', (e) => {
            this.elements.maxCpuValue.textContent = e.target.value + '%';
        });
        
        // IPC listeners
        ipcRenderer.on('status-update', (event, status) => {
            this.updateStatus(status);
        });
        
        ipcRenderer.on('show-settings', () => {
            this.showSettings();
        });
        
        // Close modal when clicking outside
        this.elements.settingsModal.addEventListener('click', (e) => {
            if (e.target === this.elements.settingsModal) {
                this.hideSettings();
            }
        });
    }
    
    async loadSettings() {
        try {
            this.settings = await ipcRenderer.invoke('get-settings');
            this.updateSettingsUI();
            
            // Set server URL in registration form
            document.getElementById('serverUrl').value = this.settings.serverUrl;
            
        } catch (error) {
            console.error('Error loading settings:', error);
            this.logActivity('Error loading settings: ' + error.message, 'error');
        }
    }
    
    updateSettingsUI() {
        this.elements.maxCpuUsage.value = this.settings.maxCpuUsage;
        this.elements.maxCpuValue.textContent = this.settings.maxCpuUsage + '%';
        this.elements.maxMemoryUsage.value = this.settings.maxMemoryUsage;
        this.elements.autoStart.checked = this.settings.autoStart;
        this.elements.runInBackground.checked = this.settings.runInBackground;
    }
    
    async handleRegistration() {
        const clientName = document.getElementById('clientName').value;
        const serverUrl = document.getElementById('serverUrl').value;
        
        if (!clientName || !serverUrl) {
            this.logActivity('Please fill in all registration fields', 'error');
            return;
        }
        
        this.logActivity('Registering client...', 'info');
        this.updateConnectionStatus('connecting', 'Registering...');
        
        try {
            // Get system info
            const systemInfo = await ipcRenderer.invoke('get-system-info');
            
            const clientData = {
                name: clientName,
                publicKey: this.generatePublicKey(),
                capabilities: systemInfo.capabilities,
                systemInfo: systemInfo
            };
            
            const result = await ipcRenderer.invoke('register-client', clientData);
            
            if (result.success) {
                this.isRegistered = true;
                this.showDashboard();
                this.updateConnectionStatus('connected', 'Connected');
                this.logActivity('Client registered successfully', 'info');
                
                // Update system info display
                this.updateSystemInfo(systemInfo);
                
            } else {
                this.updateConnectionStatus('disconnected', 'Registration failed');
                this.logActivity('Registration failed: ' + result.error, 'error');
            }
            
        } catch (error) {
            console.error('Registration error:', error);
            this.updateConnectionStatus('disconnected', 'Registration error');
            this.logActivity('Registration error: ' + error.message, 'error');
        }
    }
    
    generatePublicKey() {
        // Generate a simple public key for demonstration
        // In a real implementation, this would use proper cryptography
        return 'demo-public-key-' + Date.now();
    }
    
    showDashboard() {
        this.elements.registrationSection.classList.add('hidden');
        this.elements.dashboardSection.classList.remove('hidden');
    }
    
    updateConnectionStatus(status, text) {
        this.elements.statusIndicator.className = `status-indicator ${status}`;
        this.elements.statusText.textContent = text;
    }
    
    updateSystemInfo(systemInfo) {
        if (systemInfo.cpu) {
            this.elements.cpuInfo.textContent = 
                `${systemInfo.cpu.cores} cores @ ${systemInfo.cpu.speed || 'Unknown'} GHz`;
        }
        
        if (systemInfo.memory) {
            const memoryGB = (systemInfo.memory.total / (1024 * 1024 * 1024)).toFixed(1);
            this.elements.memoryInfo.textContent = `${memoryGB} GB`;
        }
        
        if (systemInfo.os) {
            this.elements.platformInfo.textContent = 
                `${systemInfo.os.platform} ${systemInfo.os.arch}`;
        }
    }
    
    updateStatus(status) {
        // Update connection status
        if (status.connectionStatus) {
            const connStatus = status.connectionStatus;
            if (connStatus.isConnected) {
                this.updateConnectionStatus('connected', 'Connected');
            } else {
                this.updateConnectionStatus('disconnected', 'Disconnected');
            }
        }
        
        // Update task statistics
        if (status.taskStats) {
            const stats = status.taskStats;
            this.elements.completedTasks.textContent = stats.completed || 0;
            this.elements.failedTasks.textContent = stats.failed || 0;
            this.elements.activeTasks.textContent = stats.active || 0;
            this.elements.successRate.textContent = (stats.successRate || 0).toFixed(1) + '%';
        }
        
        // Update resource usage
        if (status.systemInfo) {
            const sysInfo = status.systemInfo;
            
            if (sysInfo.cpu) {
                const cpuUsage = sysInfo.cpu.usage || 0;
                this.elements.cpuUsageFill.style.width = cpuUsage + '%';
                this.elements.cpuUsageText.textContent = cpuUsage.toFixed(1) + '%';
            }
            
            if (sysInfo.memory) {
                const memUsage = sysInfo.memory.usagePercent || 0;
                this.elements.memoryUsageFill.style.width = memUsage + '%';
                this.elements.memoryUsageText.textContent = memUsage.toFixed(1) + '%';
            }
        }
    }
    
    async toggleProcessing() {
        try {
            if (this.isPaused) {
                await ipcRenderer.invoke('resume-processing');
                this.isPaused = false;
                this.elements.pauseBtn.textContent = 'Pause Processing';
                this.elements.pauseBtn.className = 'btn btn-warning';
                this.logActivity('Processing resumed', 'info');
            } else {
                await ipcRenderer.invoke('pause-processing');
                this.isPaused = true;
                this.elements.pauseBtn.textContent = 'Resume Processing';
                this.elements.pauseBtn.className = 'btn btn-secondary';
                this.logActivity('Processing paused', 'info');
            }
        } catch (error) {
            console.error('Error toggling processing:', error);
            this.logActivity('Error toggling processing: ' + error.message, 'error');
        }
    }
    
    async requestTask() {
        try {
            this.logActivity('Requesting new task...', 'info');
            const result = await ipcRenderer.invoke('request-task');
            
            if (result) {
                this.logActivity('New task received', 'info');
            } else {
                this.logActivity('No tasks available', 'warn');
            }
        } catch (error) {
            console.error('Error requesting task:', error);
            this.logActivity('Error requesting task: ' + error.message, 'error');
        }
    }
    
    showSettings() {
        this.elements.settingsModal.classList.remove('hidden');
    }
    
    hideSettings() {
        this.elements.settingsModal.classList.add('hidden');
    }
    
    async saveSettings() {
        try {
            const newSettings = {
                maxCpuUsage: parseInt(this.elements.maxCpuUsage.value),
                maxMemoryUsage: parseInt(this.elements.maxMemoryUsage.value),
                autoStart: this.elements.autoStart.checked,
                runInBackground: this.elements.runInBackground.checked
            };
            
            await ipcRenderer.invoke('save-settings', newSettings);
            this.settings = { ...this.settings, ...newSettings };
            
            this.hideSettings();
            this.logActivity('Settings saved successfully', 'info');
            
        } catch (error) {
            console.error('Error saving settings:', error);
            this.logActivity('Error saving settings: ' + error.message, 'error');
        }
    }
    
    logActivity(message, level = 'info') {
        const timestamp = new Date().toLocaleTimeString();
        const logEntry = document.createElement('div');
        logEntry.className = 'log-entry';
        
        logEntry.innerHTML = `
            <span class="log-time">[${timestamp}]</span>
            <span class="log-level-${level}">${message}</span>
        `;
        
        this.elements.activityLog.appendChild(logEntry);
        
        // Scroll to bottom
        this.elements.activityLog.scrollTop = this.elements.activityLog.scrollHeight;
        
        // Keep only last 100 entries
        const entries = this.elements.activityLog.children;
        if (entries.length > 100) {
            this.elements.activityLog.removeChild(entries[0]);
        }
        
        console.log(`[${level.toUpperCase()}] ${message}`);
    }
    
    formatBytes(bytes) {
        if (bytes === 0) return '0 B';
        
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
}

// Initialize UI when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new ClientUI();
});
