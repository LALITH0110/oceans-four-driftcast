/**
 * VARI - Enhanced UI Controller
 */
const { ipcRenderer } = require('electron');

class OceanDriftGuardian {
    constructor() {
        this.state = {
            isRegistered: false,
            isPaused: false,
            isTurboMode: false,
            userName: 'Guest',
            userEmail: '',
            totalTasks: 0,
            totalTime: 0,
            userPoints: 0,
            achievements: [],
            currentTask: null,
            selectedAvatar: '🌊' // Default avatar
        };
        
        this.settings = {};
        this.activityChart = null;
        this.taskStartTime = null;
        this.schedule = []; // Array of active time blocks: {day: 0-6, hour: 0-23}
        this.isDragging = false; // For schedule grid dragging
        this.dragMode = null; // 'select' or 'deselect'
        
        this.initializeUI();
        this.setupEventListeners();
        this.loadSettings();
        this.startAnimations();
        this.startLiveCounters();
    }
    
    initializeUI() {
        // Cache all DOM elements
        this.elements = {
            // Sections
            welcomeSection: document.getElementById('welcomeSection'),
            dashboardSection: document.getElementById('dashboardSection'),
            authContainer: document.getElementById('authContainer'),
            
            // Forms
            registrationForm: document.getElementById('registrationForm'),
            loginForm: document.getElementById('loginForm'),
            
            // Header
            userMenu: document.getElementById('userMenu'),
            userAvatar: document.getElementById('userAvatar'),
            userName: document.getElementById('userName'),
            logoutBtn: document.getElementById('logoutBtn'),
            statusIndicator: document.getElementById('statusIndicator'),
            statusText: document.getElementById('statusText'),
            
            // Quick Stats
            userRank: document.getElementById('userRank'),
            totalTime: document.getElementById('totalTime'),
            oceanArea: document.getElementById('oceanArea'),
            achievementsCount: document.getElementById('achievements'),
            
            // Performance
            completedTasks: document.getElementById('completedTasks'),
            activeTasks: document.getElementById('activeTasks'),
            successRate: document.getElementById('successRate'),
            
            // Activity Feed
            activityFeed: document.getElementById('activityFeed'),
            currentTask: document.getElementById('currentTask'),
            taskProgress: document.getElementById('taskProgress'),
            
            // Resources
            cpuUsageFill: document.getElementById('cpuUsageFill'),
            cpuUsageText: document.getElementById('cpuUsageText'),
            memoryUsageFill: document.getElementById('memoryUsageFill'),
            memoryUsageText: document.getElementById('memoryUsageText'),
            networkActivity: document.getElementById('networkActivity'),
            networkIndicator: document.getElementById('networkIndicator'),
            platformInfo: document.getElementById('platformInfo'),
            totalMemory: document.getElementById('totalMemory'),
            
            // Controls
            pauseBtn: document.getElementById('pauseBtn'),
            turboBtn: document.getElementById('turboBtn'),
            settingsBtn: document.getElementById('settingsBtn'),
            scheduleBtn: document.getElementById('scheduleBtn'),
            
            // Modal
            settingsModal: document.getElementById('settingsModal'),
            
            // Settings
            maxCpuUsage: document.getElementById('maxCpuUsage'),
            maxCpuValue: document.getElementById('maxCpuValue'),
            maxMemoryUsage: document.getElementById('maxMemoryUsage'),
            turboMode: document.getElementById('turboMode'),
            alwaysOn: document.getElementById('alwaysOn'),
            achievementNotifs: document.getElementById('achievementNotifs'),
            milestoneNotifs: document.getElementById('milestoneNotifs'),
            leaderboardNotifs: document.getElementById('leaderboardNotifs'),

            // Battery Saver
            batteryStatus: document.getElementById('batteryStatus'),
            batteryIcon: document.getElementById('batteryIcon'),
            batteryText: document.getElementById('batteryText'),
            autoPauseOnBattery: document.getElementById('autoPauseOnBattery'),
            reduceCpuOnBattery: document.getElementById('reduceCpuOnBattery'),

            // Smart Pause
            enableSmartPause: document.getElementById('enableSmartPause'),
            smartPauseOptions: document.getElementById('smartPauseOptions'),
            smartResumeOptions: document.getElementById('smartResumeOptions'),
            pauseCpuThreshold: document.getElementById('pauseCpuThreshold'),
            pauseCpuValue: document.getElementById('pauseCpuValue'),
            resumeCpuThreshold: document.getElementById('resumeCpuThreshold'),
            resumeCpuValue: document.getElementById('resumeCpuValue'),

            // Schedule
            customScheduleSection: document.getElementById('customScheduleSection'),
            scheduleGrid: document.getElementById('scheduleGrid'),
            scheduleStatusText: document.getElementById('scheduleStatusText'),
            presetNights: document.getElementById('presetNights'),
            presetWeekends: document.getElementById('presetWeekends'),
            presetOffHours: document.getElementById('presetOffHours'),
            presetClear: document.getElementById('presetClear'),
            
            // FAB
            fabHelp: document.getElementById('fabHelp'),

            // Achievements Modal
            achievementsModal: document.getElementById('achievementsModal'),
            closeAchievements: document.getElementById('closeAchievements'),
            closeAchievementsBtn: document.getElementById('closeAchievementsBtn'),
            avatarGrid: document.getElementById('avatarGrid')
        };
        
        this.addActivity('Welcome to VARI!', 'info');
    }
    
    setupEventListeners() {
        // Auth tabs
        document.querySelectorAll('.auth-tab').forEach(tab => {
            tab.addEventListener('click', (e) => {
                this.switchAuthTab(e.target.dataset.tab);
            });
        });
        
        // Registration form
        this.elements.registrationForm.addEventListener('submit', (e) => {
            e.preventDefault();
            this.handleRegistration();
        });
        
        // Login form
        this.elements.loginForm.addEventListener('submit', (e) => {
            e.preventDefault();
            this.handleLogin();
        });
        
        // Logout
        this.elements.logoutBtn.addEventListener('click', () => {
            this.handleLogout();
        });
        
        // Controls
        this.elements.pauseBtn.addEventListener('click', () => {
            this.toggleProcessing();
        });
        
        this.elements.turboBtn.addEventListener('click', () => {
            this.toggleTurboMode();
        });
        
        this.elements.settingsBtn.addEventListener('click', () => {
            this.showSettings();
        });
        
        this.elements.scheduleBtn.addEventListener('click', () => {
            this.showSettings('schedule');
        });
        
        // Settings tabs
        document.querySelectorAll('.settings-tab').forEach(tab => {
            tab.addEventListener('click', (e) => {
                this.switchSettingsTab(e.target.dataset.section);
            });
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
        
        // Settings inputs
        this.elements.maxCpuUsage.addEventListener('input', (e) => {
            this.elements.maxCpuValue.textContent = e.target.value + '%';
        });

        // Smart Pause sliders
        this.elements.pauseCpuThreshold.addEventListener('input', (e) => {
            this.elements.pauseCpuValue.textContent = e.target.value + '%';
        });

        this.elements.resumeCpuThreshold.addEventListener('input', (e) => {
            this.elements.resumeCpuValue.textContent = e.target.value + '%';
        });

        // Smart Pause toggle
        this.elements.enableSmartPause.addEventListener('change', (e) => {
            const isEnabled = e.target.checked;
            this.elements.smartPauseOptions.style.display = isEnabled ? 'block' : 'none';
            this.elements.smartResumeOptions.style.display = isEnabled ? 'block' : 'none';
        });

        // Always On toggle
        this.elements.alwaysOn.addEventListener('change', (e) => {
            const isAlwaysOn = e.target.checked;
            this.elements.customScheduleSection.style.display = isAlwaysOn ? 'none' : 'block';
            if (!isAlwaysOn && this.schedule.length === 0) {
                this.generateScheduleGrid();
            }
        });

        // Schedule presets
        this.elements.presetNights.addEventListener('click', () => {
            this.applySchedulePreset('nights');
        });

        this.elements.presetWeekends.addEventListener('click', () => {
            this.applySchedulePreset('weekends');
        });

        this.elements.presetOffHours.addEventListener('click', () => {
            this.applySchedulePreset('offhours');
        });

        this.elements.presetClear.addEventListener('click', () => {
            this.clearSchedule();
        });
        
        // Modal backdrop click
        this.elements.settingsModal.addEventListener('click', (e) => {
            if (e.target === this.elements.settingsModal) {
                this.hideSettings();
            }
        });
        
        // Leaderboard
        document.getElementById('viewFullLeaderboard').addEventListener('click', () => {
            this.showLeaderboard();
        });
        
        // Achievements
        document.getElementById('viewAllAchievements').addEventListener('click', () => {
            this.showAchievementsModal();
        });

        this.elements.closeAchievements.addEventListener('click', () => {
            this.hideAchievementsModal();
        });

        this.elements.closeAchievementsBtn.addEventListener('click', () => {
            this.hideAchievementsModal();
        });

        // Close achievements modal on backdrop click
        this.elements.achievementsModal.addEventListener('click', (e) => {
            if (e.target === this.elements.achievementsModal) {
                this.hideAchievementsModal();
            }
        });
        
        // FAB
        this.elements.fabHelp.addEventListener('click', () => {
            this.showHelp();
        });
        
        // IPC listeners
        ipcRenderer.on('status-update', (event, status) => {
            this.updateStatus(status);
        });

        ipcRenderer.on('task-assigned', (event, task) => {
            this.handleTaskAssignment(task);
        });

        ipcRenderer.on('achievement-unlocked', (event, achievement) => {
            this.showAchievementNotification(achievement);
        });

        ipcRenderer.on('processing-paused', () => {
            this.state.isPaused = true;
            this.elements.pauseBtn.innerHTML = '<i class="fas fa-play"></i><span>Resume</span>';
            this.elements.pauseBtn.classList.add('active');
        });

        ipcRenderer.on('processing-resumed', () => {
            this.state.isPaused = false;
            this.elements.pauseBtn.innerHTML = '<i class="fas fa-pause"></i><span>Pause</span>';
            this.elements.pauseBtn.classList.remove('active');
        });

        ipcRenderer.on('battery-status-update', (event, batteryData) => {
            this.updateBatteryStatus(batteryData);
        });
    }
    
    switchAuthTab(tab) {
        document.querySelectorAll('.auth-tab').forEach(t => {
            t.classList.toggle('active', t.dataset.tab === tab);
        });
        
        const showRegister = tab === 'register';
        this.elements.registrationForm.classList.toggle('hidden', !showRegister);
        this.elements.loginForm.classList.toggle('hidden', showRegister);
    }
    
    switchSettingsTab(section) {
        document.querySelectorAll('.settings-tab').forEach(t => {
            t.classList.toggle('active', t.dataset.section === section);
        });
        
        document.getElementById('performanceSettings').classList.toggle('hidden', section !== 'performance');
        document.getElementById('scheduleSettings').classList.toggle('hidden', section !== 'schedule');
        document.getElementById('notificationSettings').classList.toggle('hidden', section !== 'notifications');
    }
    
    async loadSettings() {
        try {
            this.settings = await ipcRenderer.invoke('get-settings');
            this.updateSettingsUI();
        } catch (error) {
            console.error('Error loading settings:', error);
            this.addActivity('Error loading settings', 'error');
        }
    }
    
    updateSettingsUI() {
        this.elements.maxCpuUsage.value = this.settings.maxCpuUsage || 50;
        this.elements.maxCpuValue.textContent = (this.settings.maxCpuUsage || 50) + '%';
        this.elements.maxMemoryUsage.value = this.settings.maxMemoryUsage || 1024;
        this.elements.turboMode.checked = this.state.isTurboMode;
        this.elements.alwaysOn.checked = this.settings.alwaysOn !== false;
        this.elements.achievementNotifs.checked = this.settings.achievementNotifs !== false;
        this.elements.milestoneNotifs.checked = this.settings.milestoneNotifs !== false;
        this.elements.leaderboardNotifs.checked = this.settings.leaderboardNotifs === true;

        // Battery Saver
        this.elements.autoPauseOnBattery.checked = this.settings.autoPauseOnBattery === true;
        this.elements.reduceCpuOnBattery.checked = this.settings.reduceCpuOnBattery === true;

        // Smart Pause
        this.elements.enableSmartPause.checked = this.settings.enableSmartPause === true;
        this.elements.pauseCpuThreshold.value = this.settings.pauseCpuThreshold || 75;
        this.elements.pauseCpuValue.textContent = (this.settings.pauseCpuThreshold || 75) + '%';
        this.elements.resumeCpuThreshold.value = this.settings.resumeCpuThreshold || 40;
        this.elements.resumeCpuValue.textContent = (this.settings.resumeCpuThreshold || 40) + '%';

        // Show/hide smart pause options
        const smartPauseEnabled = this.settings.enableSmartPause === true;
        this.elements.smartPauseOptions.style.display = smartPauseEnabled ? 'block' : 'none';
        this.elements.smartResumeOptions.style.display = smartPauseEnabled ? 'block' : 'none';

        // Show/hide schedule section
        const alwaysOn = this.settings.alwaysOn !== false;
        this.elements.customScheduleSection.style.display = alwaysOn ? 'none' : 'block';

        // Load schedule
        this.schedule = this.settings.schedule || [];
        if (!alwaysOn && this.schedule.length === 0) {
            this.generateScheduleGrid();
        } else if (!alwaysOn) {
            this.generateScheduleGrid();
        }
    }
    
    async handleRegistration() {
        const clientName = document.getElementById('clientName').value.trim();
        const email = document.getElementById('email').value.trim();
        const termsAccepted = document.getElementById('termsAccept').checked;
        
        if (!clientName || !termsAccepted) {
            this.addActivity('Please fill in all required fields', 'error');
            return;
        }
        
        this.addActivity('Creating your account...', 'info');
        this.updateConnectionStatus('connecting', 'Connecting...');
        
        try {
            // Get system info
            const systemInfo = await ipcRenderer.invoke('get-system-info');
            
            const clientData = {
                name: clientName,
                email: email,
                publicKey: this.generatePublicKey(),
                capabilities: systemInfo.capabilities,
                systemInfo: systemInfo
            };
            
            const result = await ipcRenderer.invoke('register-client', clientData);
            
            if (result.success) {
                this.state.isRegistered = true;
                this.state.userName = clientName;
                this.state.userEmail = email;
                
                this.showDashboard();
                this.updateConnectionStatus('connected', 'Connected');
                this.addActivity('Welcome aboard, ' + clientName + '!', 'success');
                
                // Update UI with user info
                this.updateUserInfo();
                this.updateSystemInfo(systemInfo);
                
                // Start contribution chart
                this.initializeChart();
                
                // Show FAB
                this.elements.fabHelp.classList.remove('hidden');
                
            } else {
                this.updateConnectionStatus('disconnected', 'Failed');
                this.addActivity('Registration failed: ' + result.error, 'error');
            }
            
        } catch (error) {
            console.error('Registration error:', error);
            this.updateConnectionStatus('disconnected', 'Error');
            this.addActivity('Connection error: ' + error.message, 'error');
        }
    }
    
    async handleLogin() {
        const loginName = document.getElementById('loginName').value.trim();
        
        if (!loginName) {
            this.addActivity('Please enter your display name', 'error');
            return;
        }
        
        // Simplified login for demo - in production would verify credentials
        this.state.isRegistered = true;
        this.state.userName = loginName;
        
        this.showDashboard();
        this.updateConnectionStatus('connected', 'Connected');
        this.addActivity('Welcome back, ' + loginName + '!', 'success');
        
        this.updateUserInfo();
        this.initializeChart();
        this.elements.fabHelp.classList.remove('hidden');
    }
    
    handleLogout() {
        this.state.isRegistered = false;
        this.state.userName = 'Guest';
        this.state.userEmail = '';
        
        this.elements.welcomeSection.classList.remove('hidden');
        this.elements.dashboardSection.classList.add('hidden');
        this.elements.userMenu.classList.add('hidden');
        this.elements.fabHelp.classList.add('hidden');
        
        this.updateConnectionStatus('disconnected', 'Offline');
        this.addActivity('Logged out successfully', 'info');
    }
    
    generatePublicKey() {
        // Simple key generation for demo
        return 'PK_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }
    
    showDashboard() {
        this.elements.welcomeSection.classList.add('hidden');
        this.elements.dashboardSection.classList.remove('hidden');
        this.elements.userMenu.classList.remove('hidden');
    }
    
    updateUserInfo() {
        this.elements.userName.textContent = this.state.userName;
        this.elements.userAvatar.src = `https://ui-avatars.com/api/?name=${encodeURIComponent(this.state.userName)}&background=4A90E2&color=fff`;
        
        // Update leaderboard with user
        const userEntry = document.querySelector('.leaderboard-item.you .leaderboard-name');
        if (userEntry) {
            userEntry.textContent = this.state.userName;
        }
    }
    
    updateConnectionStatus(status, text) {
        this.elements.statusIndicator.className = `status-indicator ${status}`;
        this.elements.statusText.textContent = text;
    }
    
    updateSystemInfo(systemInfo) {
        if (systemInfo.cpu) {
            const cores = systemInfo.cpu.cores || 'Unknown';
            const speed = systemInfo.cpu.speed ? `${systemInfo.cpu.speed.toFixed(1)} GHz` : 'Unknown';
            this.elements.platformInfo.textContent = `${cores} cores @ ${speed}`;
        }
        
        if (systemInfo.memory) {
            const memoryGB = (systemInfo.memory.total / (1024 * 1024 * 1024)).toFixed(1);
            this.elements.totalMemory.textContent = `${memoryGB} GB Total RAM`;
        }
        
        if (systemInfo.os) {
            const platform = systemInfo.os.platform === 'darwin' ? 'macOS' : 
                           systemInfo.os.platform === 'win32' ? 'Windows' : 'Linux';
            this.elements.platformInfo.textContent += ` • ${platform}`;
        }
    }
    
    updateStatus(status) {
        // Connection status
        if (status.connectionStatus) {
            const connStatus = status.connectionStatus;
            if (connStatus.isConnected) {
                this.updateConnectionStatus('connected', 'Connected');
            } else {
                this.updateConnectionStatus('disconnected', 'Disconnected');
            }
        }
        
        // Task statistics
        if (status.taskStats) {
            const stats = status.taskStats;
            this.state.totalTasks = stats.completed || 0;

            this.elements.completedTasks.textContent = stats.completed || 0;
            this.elements.activeTasks.textContent = this.state.isPaused ? 0 : 1;
            this.elements.successRate.textContent = (stats.successRate || 100).toFixed(0) + '%';
            
            // Update quick stats
            this.updateQuickStats();
        }
        
        // Resource usage
        if (status.systemInfo) {
            this.updateResourceUsage(status.systemInfo);
        }
    }
    
    updateResourceUsage(sysInfo) {
        if (sysInfo.cpu) {
            // Show 0% CPU usage when paused
            const cpuUsage = this.state.isPaused ? 0 : (sysInfo.cpu.usage || 0);
            this.elements.cpuUsageFill.style.width = cpuUsage + '%';
            this.elements.cpuUsageText.textContent = cpuUsage.toFixed(0) + '%';

            // Change color based on usage
            if (cpuUsage > 80) {
                this.elements.cpuUsageFill.style.background = 'var(--error)';
            } else if (cpuUsage > 50) {
                this.elements.cpuUsageFill.style.background = 'var(--warning)';
            } else {
                // Reset to default color when usage is low
                this.elements.cpuUsageFill.style.background = '';
            }
        }

        if (sysInfo.memory) {
            const memUsage = sysInfo.memory.usagePercent || 0;
            const memUsedMB = sysInfo.memory.used ? (sysInfo.memory.used / (1024 * 1024)).toFixed(0) : 0;

            this.elements.memoryUsageFill.style.width = memUsage + '%';
            this.elements.memoryUsageText.textContent = `${memUsedMB} MB`;
        }

        // Simulate network activity - show idle when paused
        if (this.state.currentTask && !this.state.isPaused) {
            this.elements.networkActivity.textContent = 'Active';
            this.elements.networkIndicator.querySelectorAll('.network-dot').forEach((dot, i) => {
                setTimeout(() => {
                    dot.classList.add('active');
                    setTimeout(() => dot.classList.remove('active'), 500);
                }, i * 200);
            });
        } else {
            this.elements.networkActivity.textContent = 'Idle';
        }
    }
    
    updateQuickStats() {
        // Update total time (simulated)
        const hours = Math.floor(this.state.totalTime / 3600);
        const minutes = Math.floor((this.state.totalTime % 3600) / 60);
        this.elements.totalTime.textContent = hours > 0 ? `${hours}h ${minutes}m` : `${minutes}m`;
        
        // Update ocean area (simulated based on tasks)
        const areaKm2 = (this.state.totalTasks * 42.7).toFixed(0);
        this.elements.oceanArea.textContent = `${areaKm2} km²`;
        
        // Update achievements
        const achievementCount = document.querySelectorAll('.achievement.unlocked').length;
        this.elements.achievementsCount.textContent = achievementCount;
    }

    startLiveCounters() {
        // Increment computing time and user points every second
        if (this.liveInterval) return;
        this.liveInterval = setInterval(() => {
            // Only count computing time and points when NOT paused
            if (!this.state.isPaused) {
                this.state.totalTime += 1;
                this.updateQuickStats();

                // Increment personal leaderboard points like a counter
                this.state.userPoints += 1; // +1 point per second
                const youScoreEl = document.querySelector('.leaderboard-item.you .leaderboard-score');
                if (youScoreEl) {
                    youScoreEl.textContent = this.formatNumber(this.state.userPoints);
                }
            }
        }, 1000);
    }

    stopLiveCounters() {
        if (this.liveInterval) {
            clearInterval(this.liveInterval);
            this.liveInterval = null;
        }
    }

    formatNumber(n) {
        try {
            return n.toLocaleString(undefined);
        } catch (e) {
            return String(n);
        }
    }
    
    async toggleProcessing() {
        try {
            if (this.state.isPaused) {
                await ipcRenderer.invoke('resume-processing');
                this.state.isPaused = false;
                this.elements.pauseBtn.innerHTML = '<i class="fas fa-pause"></i><span>Pause</span>';
                this.elements.pauseBtn.classList.remove('active');
                this.elements.activeTasks.textContent = 1;
                this.addActivity('Processing resumed', 'success');
            } else {
                await ipcRenderer.invoke('pause-processing');
                this.state.isPaused = true;
                this.elements.pauseBtn.innerHTML = '<i class="fas fa-play"></i><span>Resume</span>';
                this.elements.pauseBtn.classList.add('active');
                this.elements.activeTasks.textContent = 0;
                this.addActivity('Processing paused', 'info');
            }
        } catch (error) {
            console.error('Error toggling processing:', error);
            this.addActivity('Error: ' + error.message, 'error');
        }
    }
    
    async toggleTurboMode() {
        this.state.isTurboMode = !this.state.isTurboMode;
        this.elements.turboBtn.classList.toggle('active', this.state.isTurboMode);

        // Send to main process to actually enable turbo mode
        try {
            await ipcRenderer.invoke('toggle-turbo-mode', this.state.isTurboMode);

            if (this.state.isTurboMode) {
                this.addActivity('Turbo mode activated! 🚀 Running 4 concurrent tasks', 'success');
                this.elements.turboBtn.style.animation = 'pulse 1s infinite';
            } else {
                this.addActivity('Turbo mode deactivated - Back to 2 tasks', 'info');
                this.elements.turboBtn.style.animation = '';
            }
        } catch (error) {
            console.error('Error toggling turbo mode:', error);
            this.addActivity('Error toggling turbo mode', 'error');
        }
    }
    
    showSettings(tab = 'performance') {
        this.elements.settingsModal.classList.remove('hidden');
        this.switchSettingsTab(tab);
    }
    
    hideSettings() {
        this.elements.settingsModal.classList.add('hidden');
    }
    
    async saveSettings() {
        try {
            const newSettings = {
                maxCpuUsage: parseInt(this.elements.maxCpuUsage.value),
                maxMemoryUsage: parseInt(this.elements.maxMemoryUsage.value),
                turboMode: this.elements.turboMode.checked,
                alwaysOn: this.elements.alwaysOn.checked,
                achievementNotifs: this.elements.achievementNotifs.checked,
                milestoneNotifs: this.elements.milestoneNotifs.checked,
                leaderboardNotifs: this.elements.leaderboardNotifs.checked,

                // Battery Saver
                autoPauseOnBattery: this.elements.autoPauseOnBattery.checked,
                reduceCpuOnBattery: this.elements.reduceCpuOnBattery.checked,

                // Smart Pause
                enableSmartPause: this.elements.enableSmartPause.checked,
                pauseCpuThreshold: parseInt(this.elements.pauseCpuThreshold.value),
                resumeCpuThreshold: parseInt(this.elements.resumeCpuThreshold.value),

                // Schedule
                schedule: this.schedule
            };

            await ipcRenderer.invoke('save-settings', newSettings);
            this.settings = { ...this.settings, ...newSettings };

            this.hideSettings();
            this.addActivity('Settings saved successfully', 'success');

        } catch (error) {
            console.error('Error saving settings:', error);
            this.addActivity('Error saving settings', 'error');
        }
    }
    
    handleTaskAssignment(task) {
        this.state.currentTask = task;
        this.taskStartTime = Date.now();
        
        const taskInfo = this.elements.currentTask.querySelector('.task-id');
        taskInfo.textContent = `#${task.task_id.substr(0, 8)}`;
        
        this.addActivity(`New task assigned: ${task.particle_count} particles`, 'info');
        
        // Simulate task progress
        let progress = 0;
        const progressInterval = setInterval(() => {
            progress += Math.random() * 15;
            if (progress >= 100) {
                progress = 100;
                clearInterval(progressInterval);
                this.completeTask();
            }
            
            this.elements.taskProgress.style.width = progress + '%';
            this.elements.currentTask.querySelector('.progress-text').textContent = 
                Math.floor(progress) + '%';
        }, 1000);
    }
    
    completeTask() {
        if (this.state.currentTask && this.taskStartTime) {
            const duration = Date.now() - this.taskStartTime;
            this.state.totalTime += duration / 1000;
            this.state.totalTasks++;
            
            this.addActivity(`Task completed in ${(duration / 1000).toFixed(1)}s`, 'success');
            
            // Reset task display
            this.state.currentTask = null;
            this.taskStartTime = null;
            this.elements.currentTask.querySelector('.task-id').textContent = 'None';
            this.elements.taskProgress.style.width = '0%';
            this.elements.currentTask.querySelector('.progress-text').textContent = '0%';
            
            // Check for achievements
            this.checkAchievements();
            
            // Update stats
            this.updateQuickStats();
        }
    }
    
    checkAchievements() {
        // First Task achievement
        if (this.state.totalTasks === 1) {
            this.unlockAchievement('first-task', 'First Task', 'fas fa-flag-checkered');
        }
        
        // 100 Tasks achievement
        if (this.state.totalTasks === 100) {
            this.unlockAchievement('100-tasks', '100 Tasks', 'fas fa-fire');
        }
        
        // 24 Hour Helper
        if (this.state.totalTime >= 86400) {
            this.unlockAchievement('24-hour', '24 Hour Helper', 'fas fa-clock');
        }
    }
    
    unlockAchievement(id, name, icon) {
        const achievement = document.querySelector(`.achievement[data-id="${id}"]`);
        if (achievement && !achievement.classList.contains('unlocked')) {
            achievement.classList.add('unlocked');
            this.showAchievementNotification({ name, icon });
        }
    }
    
    showAchievementNotification(achievement) {
        if (!this.settings.achievementNotifs) return;
        
        const notification = document.createElement('div');
        notification.className = 'achievement-notification';
        notification.innerHTML = `
            <i class="${achievement.icon}"></i>
            <div>
                <strong>Achievement Unlocked!</strong>
                <span>${achievement.name}</span>
            </div>
        `;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.classList.add('show');
        }, 100);
        
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => {
                document.body.removeChild(notification);
            }, 300);
        }, 3000);
    }
    
    addActivity(message, type = 'info') {
        const timestamp = new Date().toLocaleTimeString();
        const activityItem = document.createElement('div');
        activityItem.className = 'activity-item';
        
        const icon = type === 'error' ? 'fa-exclamation-circle' :
                    type === 'success' ? 'fa-check-circle' :
                    type === 'warning' ? 'fa-exclamation-triangle' :
                    'fa-info-circle';
        
        activityItem.innerHTML = `
            <i class="fas ${icon}"></i>
            <span>[${timestamp}] ${message}</span>
        `;
        
        this.elements.activityFeed.appendChild(activityItem);
        
        // Keep only last 50 entries
        const items = this.elements.activityFeed.children;
        if (items.length > 50) {
            this.elements.activityFeed.removeChild(items[0]);
        }
        
        // Scroll to bottom
        this.elements.activityFeed.scrollTop = this.elements.activityFeed.scrollHeight;
        
        console.log(`[${type.toUpperCase()}] ${message}`);
    }
    
    initializeChart() {
        const ctx = document.getElementById('contributionChart');
        if (!ctx) return;
        
        this.activityChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: ['6h ago', '5h ago', '4h ago', '3h ago', '2h ago', '1h ago', 'Now'],
                datasets: [{
                    label: 'Tasks Completed',
                    data: [12, 19, 15, 25, 22, 30, 0],
                    borderColor: '#4A90E2',
                    backgroundColor: 'rgba(74, 144, 226, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: {
                            color: 'rgba(0, 0, 0, 0.05)'
                        }
                    },
                    x: {
                        grid: {
                            display: false
                        }
                    }
                }
            }
        });
    }
    
    showLeaderboard() {
        this.addActivity('Full leaderboard coming soon!', 'info');
    }
    
    showAchievements() {
        this.showAchievementsModal();
    }

    showAchievementsModal() {
        this.elements.achievementsModal.classList.remove('hidden');
        this.updateAchievementProgress();
        this.generateAvatars();
        this.updateImpactMetrics();
        this.setupSocialSharing();
    }

    hideAchievementsModal() {
        this.elements.achievementsModal.classList.add('hidden');
    }

    updateAchievementProgress() {
        const totalSeconds = this.state.totalTime;
        const totalHours = totalSeconds / 3600;
        const totalTasks = this.state.totalTasks;

        // Define tier thresholds (in hours)
        const tiers = [
            { name: 'bronze', hours: 1, icon: '🥉' },
            { name: 'silver', hours: 10, icon: '🥈' },
            { name: 'gold', hours: 168, icon: '🥇' }, // 1 week
            { name: 'platinum', hours: 720, icon: '💎' }, // 1 month
            { name: 'diamond', hours: 2160, icon: '💠' } // 3 months
        ];

        tiers.forEach((tier, index) => {
            const tierElement = document.querySelector(`[data-tier="${tier.name}"]`);
            const progressFill = document.getElementById(`${tier.name}Progress`);
            const progressText = document.getElementById(`${tier.name}Text`);

            if (!tierElement || !progressFill || !progressText) return;

            const isUnlocked = totalHours >= tier.hours;
            const previousTierHours = index > 0 ? tiers[index - 1].hours : 0;
            const nextTierHours = tier.hours;

            if (isUnlocked) {
                // Unlocked
                tierElement.classList.remove('locked');
                progressFill.style.width = '100%';
                progressText.textContent = 'Unlocked ✓';
                progressText.style.color = 'var(--success)';
            } else {
                // Locked - show progress toward this tier
                tierElement.classList.add('locked');
                const progress = ((totalHours - previousTierHours) / (nextTierHours - previousTierHours)) * 100;
                const clampedProgress = Math.max(0, Math.min(100, progress));
                progressFill.style.width = clampedProgress + '%';

                const hoursRemaining = Math.max(0, nextTierHours - totalHours);
                progressText.textContent = `${hoursRemaining.toFixed(1)}h remaining`;
                progressText.style.color = 'var(--text-secondary)';
            }
        });

        // Update task achievements
        this.updateTaskAchievements(totalTasks);
    }

    updateTaskAchievements(totalTasks) {
        const taskMilestones = [
            { id: 'first-task', count: 1 },
            { id: '10-tasks', count: 10 },
            { id: '100-tasks', count: 100 },
            { id: '1000-tasks', count: 1000 }
        ];

        taskMilestones.forEach(milestone => {
            const element = document.querySelector(`[data-id="${milestone.id}"]`);
            if (!element) return;

            const isUnlocked = totalTasks >= milestone.count;

            if (isUnlocked) {
                element.classList.remove('locked');
                element.classList.add('unlocked');
                element.querySelector('.achievement-badge').textContent = '✓';
            } else {
                element.classList.add('locked');
                element.classList.remove('unlocked');
                element.querySelector('.achievement-badge').textContent = '🔒';
            }

            // Update progress text
            const progressElement = document.getElementById(`tasks${milestone.count}Progress`);
            if (progressElement) {
                progressElement.textContent = `${Math.min(totalTasks, milestone.count)}/${milestone.count}`;
            }
        });
    }

    generateAvatars() {
        const totalHours = this.state.totalTime / 3600;

        const avatars = [
            { emoji: '🌊', name: 'Wave', tier: 0 },
            { emoji: '🐠', name: 'Fish', tier: 0 },
            { emoji: '🐟', name: 'Tropical Fish', tier: 1 },
            { emoji: '🐡', name: 'Puffer', tier: 10 },
            { emoji: '🦈', name: 'Shark', tier: 10 },
            { emoji: '🐬', name: 'Dolphin', tier: 10 },
            { emoji: '🐳', name: 'Whale', tier: 168 },
            { emoji: '🐋', name: 'Blue Whale', tier: 168 },
            { emoji: '🦑', name: 'Squid', tier: 168 },
            { emoji: '🐙', name: 'Octopus', tier: 720 },
            { emoji: '🦀', name: 'Crab', tier: 720 },
            { emoji: '🦞', name: 'Lobster', tier: 720 },
            { emoji: '⚓', name: 'Anchor', tier: 2160 },
            { emoji: '🚢', name: 'Ship', tier: 2160 },
            { emoji: '⛵', name: 'Sailboat', tier: 2160 }
        ];

        const gridHTML = avatars.map(avatar => {
            const isUnlocked = totalHours >= avatar.tier;
            const isSelected = this.state.selectedAvatar === avatar.emoji;

            return `
                <div class="avatar-option ${isUnlocked ? '' : 'locked'} ${isSelected ? 'selected' : ''}"
                     data-avatar="${avatar.emoji}"
                     data-tier="${avatar.tier}">
                    ${isUnlocked ? avatar.emoji : '<span class="lock-overlay">🔒</span>'}
                </div>
            `;
        }).join('');

        this.elements.avatarGrid.innerHTML = gridHTML;

        // Add click handlers
        document.querySelectorAll('.avatar-option:not(.locked)').forEach(option => {
            option.addEventListener('click', () => {
                this.selectAvatar(option.dataset.avatar);
            });
        });
    }

    selectAvatar(emoji) {
        this.state.selectedAvatar = emoji;

        // Update avatar display
        this.elements.userAvatar.src = `https://ui-avatars.com/api/?name=${encodeURIComponent(emoji)}&background=4A90E2&color=fff&size=128`;

        // Update selection in grid
        document.querySelectorAll('.avatar-option').forEach(opt => {
            opt.classList.toggle('selected', opt.dataset.avatar === emoji);
        });

        this.addActivity(`Avatar changed to ${emoji}`, 'success');
    }

    updateImpactMetrics() {
        const totalTasks = this.state.totalTasks;
        const totalHours = this.state.totalTime / 3600;

        // Impact calculations based on research data
        // Assumptions: Each task analyzes 10 particles, each particle = ~0.1 km² coverage
        const oceanArea = (totalTasks * 10 * 0.1).toFixed(1); // km²

        // Each km² analyzed = ~0.05 tons of plastic optimized for cleanup
        const plasticSaved = (oceanArea * 0.05).toFixed(1);

        // Each hour of computing = 2 hours of manual research time saved
        const searchHoursSaved = (totalHours * 2).toFixed(0);

        // Average beach = 1 km², so beaches = ocean area
        const beaches = Math.floor(oceanArea);

        // Update DOM
        document.getElementById('impactOceanArea').textContent = this.formatNumber(oceanArea);
        document.getElementById('impactPlasticSaved').textContent = this.formatNumber(plasticSaved);
        document.getElementById('impactSearchHours').textContent = this.formatNumber(searchHoursSaved);
        document.getElementById('impactBeaches').textContent = this.formatNumber(beaches);

        // Update share card
        this.updateShareCard(oceanArea, totalTasks, totalHours);
    }

    updateShareCard(oceanArea, totalTasks, totalHours) {
        document.getElementById('shareMainStat').textContent = `${oceanArea} km²`;

        const messages = [
            `I helped analyze ocean plastic drift patterns!`,
            `I'm making waves in ocean conservation!`,
            `Contributing to cleaner oceans, one task at a time!`,
            `My computer is helping save our oceans!`,
            `Join me in the fight against ocean plastic!`
        ];
        const randomMessage = messages[Math.floor(Math.random() * messages.length)];
        document.getElementById('shareMessage').textContent = randomMessage;

        document.getElementById('shareSubstat1').textContent = `${totalTasks} tasks completed`;
        document.getElementById('shareSubstat2').textContent = `${totalHours.toFixed(1)} hours contributed`;
    }

    setupSocialSharing() {
        const oceanArea = (this.state.totalTasks * 10 * 0.1).toFixed(1);
        const totalHours = (this.state.totalTime / 3600).toFixed(1);

        // Twitter share
        const twitterBtn = document.getElementById('shareTwitter');
        if (twitterBtn) {
            twitterBtn.onclick = () => {
                const text = `🌊 I helped analyze ${oceanArea} km² of ocean to fight plastic pollution! Join me at DriftCast to make an impact! #OceanConservation #CleanOceans`;
                const url = `https://twitter.com/intent/tweet?text=${encodeURIComponent(text)}&url=${encodeURIComponent('https://driftcast.org')}`;
                require('electron').shell.openExternal(url);
                this.addActivity('Shared on Twitter!', 'success');
            };
        }

        // LinkedIn share
        const linkedinBtn = document.getElementById('shareLinkedIn');
        if (linkedinBtn) {
            linkedinBtn.onclick = () => {
                const url = `https://www.linkedin.com/sharing/share-offsite/?url=${encodeURIComponent('https://driftcast.org')}`;
                require('electron').shell.openExternal(url);
                this.addActivity('Shared on LinkedIn!', 'success');
            };
        }

        // Facebook share
        const facebookBtn = document.getElementById('shareFacebook');
        if (facebookBtn) {
            facebookBtn.onclick = () => {
                const url = `https://www.facebook.com/sharer/sharer.php?u=${encodeURIComponent('https://driftcast.org')}`;
                require('electron').shell.openExternal(url);
                this.addActivity('Shared on Facebook!', 'success');
            };
        }

        // Download card
        const downloadBtn = document.getElementById('downloadCard');
        if (downloadBtn) {
            downloadBtn.onclick = () => {
                this.downloadShareCard();
            };
        }

        // Copy referral link
        const copyBtn = document.getElementById('copyReferralLink');
        if (copyBtn) {
            copyBtn.onclick = () => {
                const input = document.getElementById('referralLinkInput');
                input.select();
                document.execCommand('copy');
                this.addActivity('Referral link copied!', 'success');
            };
        }
    }

    downloadShareCard() {
        // This would use html2canvas or similar library in production
        // For now, we'll show a message
        this.addActivity('Share card download coming soon! For now, take a screenshot 📸', 'info');

        // In production, you'd do something like:
        // const card = document.querySelector('.share-card');
        // html2canvas(card).then(canvas => {
        //     const link = document.createElement('a');
        //     link.download = 'driftcast-impact.png';
        //     link.href = canvas.toDataURL();
        //     link.click();
        // });
    }
    
    showHelp() {
        this.addActivity('Help center coming soon!', 'info');
    }
    
    startAnimations() {
        // Animate stats on dashboard load
        if (this.state.isRegistered) {
            setTimeout(() => {
                document.querySelectorAll('.quick-stat').forEach((stat, i) => {
                    setTimeout(() => {
                        stat.style.animation = 'fadeIn 0.5s ease-out';
                    }, i * 100);
                });
            }, 100);
        }
    }

    // Battery Status Update
    updateBatteryStatus(batteryData) {
        if (!batteryData) return;

        const { hasBattery, isCharging, percent, acConnected } = batteryData;
        const batteryEl = this.elements.batteryStatus;
        const iconEl = this.elements.batteryIcon;
        const textEl = this.elements.batteryText;

        // Remove existing status classes
        batteryEl.classList.remove('charging', 'on-battery', 'low-battery');

        if (!hasBattery) {
            iconEl.className = 'fas fa-plug';
            textEl.textContent = 'AC Power (Desktop)';
            batteryEl.classList.add('charging');
        } else if (isCharging || acConnected) {
            iconEl.className = 'fas fa-charging-station';
            textEl.textContent = `Charging ${percent}%`;
            batteryEl.classList.add('charging');
        } else {
            iconEl.className = 'fas fa-battery-three-quarters';
            textEl.textContent = `On Battery ${percent}%`;
            batteryEl.classList.add('on-battery');

            if (percent < 20) {
                batteryEl.classList.add('low-battery');
                iconEl.className = 'fas fa-battery-quarter';
            }
        }
    }

    // Generate Schedule Grid
    generateScheduleGrid() {
        const days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
        const gridContainer = this.elements.scheduleGrid;

        let gridHTML = '<div class="schedule-grid-wrapper">';

        // Header row with hour labels
        gridHTML += '<div class="schedule-grid-header">';
        gridHTML += '<div class="schedule-day-label"></div>'; // Empty corner
        for (let hour = 0; hour < 24; hour++) {
            gridHTML += `<div class="schedule-hour-label">${hour}</div>`;
        }
        gridHTML += '</div>';

        // Day rows
        for (let day = 0; day < 7; day++) {
            gridHTML += '<div class="schedule-grid-row">';
            gridHTML += `<div class="schedule-day-label">${days[day]}</div>`;

            for (let hour = 0; hour < 24; hour++) {
                const isActive = this.isScheduleSlotActive(day, hour);
                gridHTML += `<div class="schedule-cell ${isActive ? 'active' : ''}"
                                  data-day="${day}" data-hour="${hour}"></div>`;
            }
            gridHTML += '</div>';
        }

        gridHTML += '</div>';
        gridContainer.innerHTML = gridHTML;

        // Add mouse event listeners for dragging
        const cells = gridContainer.querySelectorAll('.schedule-cell');
        cells.forEach(cell => {
            cell.addEventListener('mousedown', (e) => {
                e.preventDefault();
                this.isDragging = true;
                this.dragMode = cell.classList.contains('active') ? 'deselect' : 'select';
                this.toggleScheduleCell(cell);
            });

            cell.addEventListener('mouseenter', () => {
                if (this.isDragging) {
                    this.toggleScheduleCell(cell);
                }
            });

            cell.addEventListener('mouseup', () => {
                this.isDragging = false;
            });
        });

        document.addEventListener('mouseup', () => {
            this.isDragging = false;
        });
    }

    isScheduleSlotActive(day, hour) {
        return this.schedule.some(slot => slot.day === day && slot.hour === hour);
    }

    toggleScheduleCell(cell) {
        const day = parseInt(cell.dataset.day);
        const hour = parseInt(cell.dataset.hour);

        if (this.dragMode === 'select') {
            // Add to schedule
            if (!this.isScheduleSlotActive(day, hour)) {
                this.schedule.push({ day, hour });
                cell.classList.add('active');
            }
        } else {
            // Remove from schedule
            this.schedule = this.schedule.filter(slot => !(slot.day === day && slot.hour === hour));
            cell.classList.remove('active');
        }
    }

    applySchedulePreset(preset) {
        this.schedule = [];

        switch (preset) {
            case 'nights':
                // 10 PM to 6 AM every day
                for (let day = 0; day < 7; day++) {
                    for (let hour = 22; hour < 24; hour++) {
                        this.schedule.push({ day, hour });
                    }
                    for (let hour = 0; hour < 6; hour++) {
                        this.schedule.push({ day, hour });
                    }
                }
                break;

            case 'weekends':
                // All day Saturday (6) and Sunday (0)
                for (let day of [0, 6]) {
                    for (let hour = 0; hour < 24; hour++) {
                        this.schedule.push({ day, hour });
                    }
                }
                break;

            case 'offhours':
                // Weekdays 6 PM to 9 AM, plus all weekend
                for (let day = 1; day < 6; day++) { // Mon-Fri
                    for (let hour = 18; hour < 24; hour++) {
                        this.schedule.push({ day, hour });
                    }
                    for (let hour = 0; hour < 9; hour++) {
                        this.schedule.push({ day, hour });
                    }
                }
                // Add full weekend
                for (let day of [0, 6]) {
                    for (let hour = 0; hour < 24; hour++) {
                        this.schedule.push({ day, hour });
                    }
                }
                break;
        }

        this.generateScheduleGrid();
        this.addActivity(`Applied ${preset} schedule preset`, 'info');
    }

    clearSchedule() {
        this.schedule = [];
        this.generateScheduleGrid();
        this.addActivity('Schedule cleared', 'info');
    }

    isTimeInSchedule() {
        const now = new Date();
        const day = now.getDay();
        const hour = now.getHours();

        return this.isScheduleSlotActive(day, hour);
    }
}

// Add achievement notification styles
const style = document.createElement('style');
style.textContent = `
.achievement-notification {
    position: fixed;
    top: 2rem;
    right: 2rem;
    background: var(--white);
    padding: 1rem 1.5rem;
    border-radius: 12px;
    box-shadow: var(--shadow-lg);
    display: flex;
    align-items: center;
    gap: 1rem;
    transform: translateX(400px);
    transition: transform 0.3s ease;
    z-index: 1001;
}

.achievement-notification.show {
    transform: translateX(0);
}

.achievement-notification i {
    font-size: 2rem;
    color: var(--success);
}

.achievement-notification strong {
    display: block;
    color: var(--text-primary);
    margin-bottom: 0.25rem;
}

.achievement-notification span {
    color: var(--text-secondary);
    font-size: 0.875rem;
}
`;
document.head.appendChild(style);

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new OceanDriftGuardian();
});