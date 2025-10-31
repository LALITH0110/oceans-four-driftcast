/**
 * Main Electron process for Ocean Plastic Forecast Client
 */
const { app, BrowserWindow, ipcMain, Menu, Tray, dialog } = require('electron');
const path = require('path');
const log = require('electron-log');
const Store = require('electron-store');

const TaskManager = require('./task-manager');
const ServerCommunicator = require('./server-communicator');
const SystemMonitor = require('./system-monitor');

// Configure logging
log.transports.file.level = 'info';
log.transports.console.level = 'debug';

// Initialize store for persistent settings
const store = new Store();

class OceanForecastApp {
    constructor() {
        this.mainWindow = null;
        this.tray = null;
        this.serverCommunicator = new ServerCommunicator();
        this.taskManager = new TaskManager(this.serverCommunicator);
        this.systemMonitor = new SystemMonitor();

        this.isQuitting = false;
        this.isRegistered = false;
        this.autoPausedReason = null; // Track if we auto-paused for smart features
    }
    
    async initialize() {
        log.info('Initializing Ocean Plastic Forecast Client...');
        
        try {
            // Create main window
            this.createWindow();
            
            // Create system tray
            this.createTray();
            
            // Setup IPC handlers
            this.setupIPC();
            
            // Initialize components
            await this.initializeComponents();
            
            log.info('Application initialized successfully');
            
        } catch (error) {
            log.error('Error during initialization:', error);
            dialog.showErrorBox('Initialization Error', error.message);
        }
    }
    
    createWindow() {
        this.mainWindow = new BrowserWindow({
            width: 1000,
            height: 700,
            webPreferences: {
                nodeIntegration: true,
                contextIsolation: false,
                enableRemoteModule: true
            },
            // icon: path.join(__dirname, '../../assets/icon.png'), // Commented out until icon is available
            show: false
        });
        
        // Load the UI
        this.mainWindow.loadFile(path.join(__dirname, '../renderer/index.html'));
        
        // Handle window events
        this.mainWindow.on('ready-to-show', () => {
            this.mainWindow.show();
        });
        
        this.mainWindow.on('close', (event) => {
            if (!this.isQuitting) {
                event.preventDefault();
                this.mainWindow.hide();
                
                // Show notification about running in background
                if (process.platform === 'darwin') {
                    app.dock.hide();
                }
            }
        });
        
        this.mainWindow.on('closed', () => {
            this.mainWindow = null;
        });
        
        // Open DevTools in development
        if (process.env.NODE_ENV === 'development') {
            this.mainWindow.webContents.openDevTools();
        }
    }
    
    createTray() {
        // Try to use icon, fallback to empty tray if not found
        const iconPath = path.join(__dirname, '../../assets/tray-icon.png');
        try {
            this.tray = new Tray(iconPath);
        } catch (error) {
            log.warn('Tray icon not found, creating tray without icon');
            // Create a simple 16x16 transparent icon programmatically
            const { nativeImage } = require('electron');
            const emptyIcon = nativeImage.createEmpty();
            this.tray = new Tray(emptyIcon);
        }
        
        const contextMenu = Menu.buildFromTemplate([
            {
                label: 'Show App',
                click: () => {
                    this.mainWindow.show();
                    if (process.platform === 'darwin') {
                        app.dock.show();
                    }
                }
            },
            {
                label: 'Task Status',
                click: () => {
                    this.showTaskStatus();
                }
            },
            { type: 'separator' },
            {
                label: 'Settings',
                click: () => {
                    this.showSettings();
                }
            },
            { type: 'separator' },
            {
                label: 'Quit',
                click: () => {
                    this.isQuitting = true;
                    app.quit();
                }
            }
        ]);
        
        this.tray.setContextMenu(contextMenu);
        this.tray.setToolTip('Ocean Plastic Forecast Client');
        
        this.tray.on('double-click', () => {
            this.mainWindow.show();
        });
    }
    
    setupIPC() {
        // Handle registration
        ipcMain.handle('register-client', async (event, clientData) => {
            try {
                const result = await this.serverCommunicator.registerClient(clientData);
                if (result.success) {
                    this.isRegistered = true;
                    store.set('clientId', result.clientId);
                    store.set('token', result.token);
                    
                    // Start task processing
                    await this.taskManager.start();
                }
                return result;
            } catch (error) {
                log.error('Registration error:', error);
                return { success: false, error: error.message };
            }
        });
        
        // Handle settings
        ipcMain.handle('get-settings', () => {
            return {
                // Production server
                serverUrl: store.get('serverUrl', 'https://system76.rice.iit.edu'),
                // Local development (uncomment for local testing)
                // serverUrl: store.get('serverUrl', 'http://localhost:8000'),
                maxCpuUsage: store.get('maxCpuUsage', 50),
                maxMemoryUsage: store.get('maxMemoryUsage', 1024),
                autoStart: store.get('autoStart', false),
                runInBackground: store.get('runInBackground', true)
            };
        });
        
        ipcMain.handle('save-settings', (event, settings) => {
            Object.keys(settings).forEach(key => {
                store.set(key, settings[key]);
            });
            return { success: true };
        });
        
        // Handle system info
        ipcMain.handle('get-system-info', async () => {
            return await this.systemMonitor.getSystemInfo();
        });
        
        // Handle task statistics
        ipcMain.handle('get-task-stats', () => {
            return this.taskManager.getStatistics();
        });
        
        // Handle connection status
        ipcMain.handle('get-connection-status', () => {
            return this.serverCommunicator.getConnectionStatus();
        });
        
        // Handle manual task request
        ipcMain.handle('request-task', async () => {
            return await this.taskManager.requestTask();
        });
        
        // Handle pause/resume
        ipcMain.handle('pause-processing', async () => {
            await this.taskManager.pause();
            // Notify renderer about pause state
            if (this.mainWindow && !this.mainWindow.isDestroyed()) {
                this.mainWindow.webContents.send('processing-paused');
            }
            return { success: true };
        });

        ipcMain.handle('resume-processing', () => {
            this.taskManager.resume();
            // Notify renderer about resume state
            if (this.mainWindow && !this.mainWindow.isDestroyed()) {
                this.mainWindow.webContents.send('processing-resumed');
            }
            return { success: true };
        });

        // Handle turbo mode toggle
        ipcMain.handle('toggle-turbo-mode', (event, enabled) => {
            this.taskManager.setTurboMode(enabled);
            return { success: true };
        });
    }
    
    async initializeComponents() {
        // Check if already registered
        const clientId = store.get('clientId');
        const token = store.get('token');
        
        if (clientId && token) {
            this.serverCommunicator.setCredentials(clientId, token);
            this.isRegistered = true;
            
            // Verify token is still valid
            const isValid = await this.serverCommunicator.verifyToken();
            if (isValid) {
                await this.taskManager.start();
                log.info('Resumed with existing credentials');
            } else {
                // Token expired, need to re-register
                this.isRegistered = false;
                store.delete('clientId');
                store.delete('token');
                log.info('Token expired, re-registration required');
            }
        }
        
        // Auto-register if not registered yet
        if (!this.isRegistered) {
            try {
                log.info('No existing credentials. Attempting auto-registration...');
                const systemInfo = await this.systemMonitor.getSystemInfo();
                const autoClientData = {
                    name: `Client-${require('os').hostname()}`,
                    publicKey: `PK_${Date.now()}_${Math.random().toString(36).slice(2, 10)}`,
                    capabilities: systemInfo?.capabilities || {},
                    systemInfo: systemInfo || {}
                };
                const result = await this.serverCommunicator.registerClient(autoClientData);
                if (result?.success) {
                    this.isRegistered = true;
                    store.set('clientId', result.clientId);
                    store.set('token', result.token);
                    await this.taskManager.start();
                    log.info('Auto-registration succeeded, task manager started');
                } else {
                    log.warn(`Auto-registration failed: ${result?.error || 'unknown error'}`);
                }
            } catch (e) {
                log.error('Auto-registration error:', e);
            }
        }
        
        // Start system monitoring
        this.systemMonitor.start();
        
        // Setup periodic status updates
        setInterval(() => {
            this.updateUI();
        }, 5000); // Update every 5 seconds

        // Setup smart pause checker
        setInterval(() => {
            this.checkSmartPause();
        }, 10000); // Check every 10 seconds
    }

    async checkSmartPause() {
        const settings = store.get();
        const systemStats = this.systemMonitor.getCurrentStats();

        if (!systemStats) return;

        let shouldPause = false;
        let pauseReason = '';

        // Check 1: Battery-based pause
        if (settings.autoPauseOnBattery && systemStats.battery) {
            const { hasBattery, isCharging, acConnected } = systemStats.battery;
            if (hasBattery && !isCharging && !acConnected) {
                shouldPause = true;
                pauseReason = 'on battery power';
            }
        }

        // Check 2: Schedule-based pause
        if (settings.schedule && settings.schedule.length > 0 && !settings.alwaysOn) {
            const now = new Date();
            const currentDay = now.getDay();
            const currentHour = now.getHours();

            const isInSchedule = settings.schedule.some(slot =>
                slot.day === currentDay && slot.hour === currentHour
            );

            if (!isInSchedule) {
                shouldPause = true;
                pauseReason = 'outside scheduled hours';
            }
        }

        // Check 3: Smart pause based on system CPU
        if (settings.enableSmartPause && systemStats.cpu) {
            const systemCpu = systemStats.cpu.usage || 0;
            const pauseThreshold = settings.pauseCpuThreshold || 75;

            // Only pause if system CPU is high
            if (systemCpu > pauseThreshold && !this.taskManager.isPaused) {
                shouldPause = true;
                pauseReason = `system CPU at ${systemCpu.toFixed(0)}%`;
            }

            // Resume if system CPU is low and we were auto-paused
            const resumeThreshold = settings.resumeCpuThreshold || 40;
            if (systemCpu < resumeThreshold && this.taskManager.isPaused && this.autoPausedReason) {
                shouldPause = false;
                pauseReason = '';
            }
        }

        // Apply pause/resume
        if (shouldPause && !this.taskManager.isPaused) {
            log.info(`Auto-pausing: ${pauseReason}`);
            this.autoPausedReason = pauseReason;
            await this.taskManager.pause();

            // Notify renderer
            if (this.mainWindow && !this.mainWindow.isDestroyed()) {
                this.mainWindow.webContents.send('processing-paused');
            }
        } else if (!shouldPause && this.taskManager.isPaused && this.autoPausedReason) {
            log.info('Auto-resuming: conditions cleared');
            this.autoPausedReason = null;
            this.taskManager.resume();

            // Notify renderer
            if (this.mainWindow && !this.mainWindow.isDestroyed()) {
                this.mainWindow.webContents.send('processing-resumed');
            }
        }
    }
    
    updateUI() {
        if (this.mainWindow && !this.mainWindow.isDestroyed()) {
            const status = {
                isRegistered: this.isRegistered,
                connectionStatus: this.serverCommunicator.getConnectionStatus(),
                taskStats: this.taskManager.getStatistics(),
                systemInfo: this.systemMonitor.getCurrentStats()
            };

            this.mainWindow.webContents.send('status-update', status);

            // Send battery status separately
            const batteryData = this.systemMonitor.getCurrentStats().battery;
            if (batteryData) {
                this.mainWindow.webContents.send('battery-status-update', batteryData);
            }
        }
    }
    
    showTaskStatus() {
        const stats = this.taskManager.getStatistics();
        const message = `
Tasks Completed: ${stats.completed}
Tasks Failed: ${stats.failed}
Currently Processing: ${stats.active}
Success Rate: ${stats.successRate.toFixed(1)}%
        `.trim();
        
        dialog.showMessageBox(this.mainWindow, {
            type: 'info',
            title: 'Task Status',
            message: message
        });
    }
    
    showSettings() {
        if (this.mainWindow) {
            this.mainWindow.show();
            this.mainWindow.webContents.send('show-settings');
        }
    }
    
    async shutdown() {
        log.info('Shutting down application...');
        
        try {
            // Stop task processing
            if (this.taskManager) {
                await this.taskManager.stop();
            }
            
            // Disconnect from server
            if (this.serverCommunicator) {
                await this.serverCommunicator.disconnect();
            }
            
            // Stop system monitoring
            if (this.systemMonitor) {
                this.systemMonitor.stop();
            }
            
            log.info('Application shutdown complete');
            
        } catch (error) {
            log.error('Error during shutdown:', error);
        }
    }
}

// Create app instance
const oceanApp = new OceanForecastApp();

// App event handlers
app.whenReady().then(() => {
    oceanApp.initialize();
});

app.on('window-all-closed', () => {
    // Keep app running in background on all platforms
    // Don't quit unless explicitly requested
});

app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
        oceanApp.createWindow();
    }
});

app.on('before-quit', async (event) => {
    if (!oceanApp.isQuitting) {
        event.preventDefault();
        oceanApp.isQuitting = true;
        
        await oceanApp.shutdown();
        app.quit();
    }
});

// Handle uncaught exceptions
process.on('uncaughtException', (error) => {
    log.error('Uncaught exception:', error);
});

process.on('unhandledRejection', (reason, promise) => {
    log.error('Unhandled rejection at:', promise, 'reason:', reason);
});
