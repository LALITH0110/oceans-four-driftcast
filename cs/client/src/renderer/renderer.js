/**
 * Ocean Drift Guardian - Enhanced UI Controller
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
            currentTask: null
        };
        
        this.settings = {};
        this.activityChart = null;
        this.taskStartTime = null;
        
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
            
            // FAB
            fabHelp: document.getElementById('fabHelp')
        };
        
        this.addActivity('Welcome to Ocean Drift Guardian!', 'info');
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
            this.showAchievements();
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
            this.elements.activeTasks.textContent = stats.active || 0;
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
            const cpuUsage = sysInfo.cpu.usage || 0;
            this.elements.cpuUsageFill.style.width = cpuUsage + '%';
            this.elements.cpuUsageText.textContent = cpuUsage.toFixed(0) + '%';
            
            // Change color based on usage
            if (cpuUsage > 80) {
                this.elements.cpuUsageFill.style.background = 'var(--error)';
            } else if (cpuUsage > 50) {
                this.elements.cpuUsageFill.style.background = 'var(--warning)';
            }
        }
        
        if (sysInfo.memory) {
            const memUsage = sysInfo.memory.usagePercent || 0;
            const memUsedMB = sysInfo.memory.used ? (sysInfo.memory.used / (1024 * 1024)).toFixed(0) : 0;
            
            this.elements.memoryUsageFill.style.width = memUsage + '%';
            this.elements.memoryUsageText.textContent = `${memUsedMB} MB`;
        }
        
        // Simulate network activity
        if (this.state.currentTask) {
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
            // Always count computing time while app is open
            this.state.totalTime += 1;
            this.updateQuickStats();

            // Increment personal leaderboard points like a counter
            this.state.userPoints += 1; // +1 point per second
            const youScoreEl = document.querySelector('.leaderboard-item.you .leaderboard-score');
            if (youScoreEl) {
                youScoreEl.textContent = this.formatNumber(this.state.userPoints);
            }
        }, 1000);
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
                this.addActivity('Processing resumed', 'success');
            } else {
                await ipcRenderer.invoke('pause-processing');
                this.state.isPaused = true;
                this.elements.pauseBtn.innerHTML = '<i class="fas fa-play"></i><span>Resume</span>';
                this.elements.pauseBtn.classList.add('active');
                this.addActivity('Processing paused', 'info');
            }
        } catch (error) {
            console.error('Error toggling processing:', error);
            this.addActivity('Error: ' + error.message, 'error');
        }
    }
    
    toggleTurboMode() {
        this.state.isTurboMode = !this.state.isTurboMode;
        this.elements.turboBtn.classList.toggle('active', this.state.isTurboMode);
        
        if (this.state.isTurboMode) {
            this.addActivity('Turbo mode activated! 🚀', 'success');
            this.elements.turboBtn.style.animation = 'pulse 1s infinite';
        } else {
            this.addActivity('Turbo mode deactivated', 'info');
            this.elements.turboBtn.style.animation = '';
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
                leaderboardNotifs: this.elements.leaderboardNotifs.checked
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
        this.addActivity('Achievements gallery coming soon!', 'info');
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