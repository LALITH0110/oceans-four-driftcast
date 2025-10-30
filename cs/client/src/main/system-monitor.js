/**
 * System monitoring for resource usage and capabilities
 */
const si = require('systeminformation');
const os = require('os');
const log = require('electron-log');

class SystemMonitor {
    constructor() {
        this.isRunning = false;
        this.monitoringInterval = null;
        this.batteryInterval = null;
        this.currentStats = {
            cpu: { usage: 0, cores: 0 },
            memory: { used: 0, total: 0, available: 0 },
            disk: { used: 0, total: 0 },
            network: { rx: 0, tx: 0 },
            battery: { hasBattery: false, isCharging: false, percent: 100, timeRemaining: null },
            timestamp: new Date()
        };

        this.systemInfo = null;
        this.updateInterval = 5000; // 5 seconds
        this.batteryUpdateInterval = 30000; // 30 seconds for battery
    }
    
    async start() {
        if (this.isRunning) {
            return;
        }
        
        log.info('Starting system monitor...');
        this.isRunning = true;
        
        // Get static system information
        await this.getSystemInfo();
        
        // Start periodic monitoring
        this.monitoringInterval = setInterval(async () => {
            await this.updateStats();
        }, this.updateInterval);

        // Start battery monitoring
        this.batteryInterval = setInterval(async () => {
            await this.updateBatteryStatus();
        }, this.batteryUpdateInterval);

        // Initial stats update
        await this.updateStats();
        await this.updateBatteryStatus();

        log.info('System monitor started');
    }
    
    stop() {
        if (!this.isRunning) {
            return;
        }

        log.info('Stopping system monitor...');
        this.isRunning = false;

        if (this.monitoringInterval) {
            clearInterval(this.monitoringInterval);
            this.monitoringInterval = null;
        }

        if (this.batteryInterval) {
            clearInterval(this.batteryInterval);
            this.batteryInterval = null;
        }

        log.info('System monitor stopped');
    }
    
    async getSystemInfo() {
        try {
            const [cpu, mem, osInfo, graphics] = await Promise.all([
                si.cpu(),
                si.mem(),
                si.osInfo(),
                si.graphics()
            ]);
            
            this.systemInfo = {
                cpu: {
                    manufacturer: cpu.manufacturer,
                    brand: cpu.brand,
                    cores: cpu.cores,
                    physicalCores: cpu.physicalCores,
                    speed: cpu.speed,
                    speedMax: cpu.speedmax,
                    cache: cpu.cache
                },
                memory: {
                    total: mem.total,
                    type: 'Unknown' // Would need additional detection
                },
                os: {
                    platform: osInfo.platform,
                    distro: osInfo.distro,
                    release: osInfo.release,
                    arch: osInfo.arch,
                    hostname: osInfo.hostname
                },
                graphics: graphics.controllers.map(gpu => ({
                    vendor: gpu.vendor,
                    model: gpu.model,
                    vram: gpu.vram,
                    bus: gpu.bus
                })),
                node: {
                    version: process.version,
                    arch: process.arch,
                    platform: process.platform
                }
            };
            
            log.info('System information collected');
            
        } catch (error) {
            log.error('Error collecting system information:', error);
            
            // Fallback to basic Node.js info
            this.systemInfo = {
                cpu: {
                    cores: os.cpus().length,
                    model: os.cpus()[0]?.model || 'Unknown'
                },
                memory: {
                    total: os.totalmem()
                },
                os: {
                    platform: os.platform(),
                    release: os.release(),
                    arch: os.arch(),
                    hostname: os.hostname()
                },
                node: {
                    version: process.version,
                    arch: process.arch,
                    platform: process.platform
                }
            };
        }
        
        return this.systemInfo;
    }
    
    async updateStats() {
        try {
            const [cpuLoad, memory, diskIO, networkStats] = await Promise.all([
                si.currentLoad(),
                si.mem(),
                si.disksIO(),
                si.networkStats()
            ]);
            
            this.currentStats = {
                cpu: {
                    usage: cpuLoad.currentLoad,
                    cores: cpuLoad.cpus.length,
                    loadAvg: os.loadavg()
                },
                memory: {
                    used: memory.used,
                    total: memory.total,
                    available: memory.available,
                    usagePercent: (memory.used / memory.total) * 100
                },
                disk: {
                    readBytes: diskIO.rIO_sec || 0,
                    writeBytes: diskIO.wIO_sec || 0,
                    readOps: diskIO.rIO || 0,
                    writeOps: diskIO.wIO || 0
                },
                network: {
                    rx: networkStats[0]?.rx_sec || 0,
                    tx: networkStats[0]?.tx_sec || 0,
                    rxTotal: networkStats[0]?.rx_bytes || 0,
                    txTotal: networkStats[0]?.tx_bytes || 0
                },
                timestamp: new Date()
            };
            
        } catch (error) {
            log.error('Error updating system stats:', error);
            
            // Fallback to basic stats
            this.currentStats = {
                cpu: {
                    usage: 0,
                    cores: os.cpus().length,
                    loadAvg: os.loadavg()
                },
                memory: {
                    used: os.totalmem() - os.freemem(),
                    total: os.totalmem(),
                    available: os.freemem(),
                    usagePercent: ((os.totalmem() - os.freemem()) / os.totalmem()) * 100
                },
                disk: { readBytes: 0, writeBytes: 0, readOps: 0, writeOps: 0 },
                network: { rx: 0, tx: 0, rxTotal: 0, txTotal: 0 },
                timestamp: new Date()
            };
        }
    }

    async updateBatteryStatus() {
        try {
            const battery = await si.battery();

            this.currentStats.battery = {
                hasBattery: battery.hasBattery,
                isCharging: battery.isCharging,
                percent: battery.percent || 100,
                timeRemaining: battery.timeRemaining || null,
                acConnected: battery.acConnected,
                type: battery.type || 'Unknown',
                model: battery.model || 'Unknown'
            };

            log.debug(`Battery status: ${battery.isCharging ? 'Charging' : 'On Battery'} (${battery.percent}%)`);

        } catch (error) {
            log.debug('Battery information not available:', error.message);

            // Fallback - assume desktop/no battery
            this.currentStats.battery = {
                hasBattery: false,
                isCharging: true,
                percent: 100,
                timeRemaining: null,
                acConnected: true,
                type: 'AC Power',
                model: 'Desktop'
            };
        }
    }

    getCurrentStats() {
        return { ...this.currentStats };
    }
    
    getCapabilities() {
        if (!this.systemInfo) {
            return {
                cpu_cores: os.cpus().length,
                memory_mb: Math.floor(os.totalmem() / (1024 * 1024)),
                platform: os.platform(),
                arch: os.arch()
            };
        }
        
        return {
            cpu_cores: this.systemInfo.cpu.cores,
            cpu_speed: this.systemInfo.cpu.speed,
            memory_mb: Math.floor(this.systemInfo.memory.total / (1024 * 1024)),
            platform: this.systemInfo.os.platform,
            arch: this.systemInfo.os.arch,
            gpu_count: this.systemInfo.graphics?.length || 0,
            node_version: this.systemInfo.node.version
        };
    }
    
    isResourceAvailable(cpuThreshold = 80, memoryThreshold = 80) {
        const stats = this.getCurrentStats();
        
        const cpuAvailable = stats.cpu.usage < cpuThreshold;
        const memoryAvailable = stats.memory.usagePercent < memoryThreshold;
        
        return {
            available: cpuAvailable && memoryAvailable,
            cpu: {
                usage: stats.cpu.usage,
                available: cpuAvailable,
                threshold: cpuThreshold
            },
            memory: {
                usage: stats.memory.usagePercent,
                available: memoryAvailable,
                threshold: memoryThreshold
            }
        };
    }
    
    getResourceLimits() {
        const capabilities = this.getCapabilities();
        
        return {
            maxCpuCores: Math.max(1, Math.floor(capabilities.cpu_cores * 0.5)), // Use up to 50% of cores
            maxMemoryMB: Math.max(512, Math.floor(capabilities.memory_mb * 0.3)), // Use up to 30% of memory
            maxDiskMB: 1000, // 1GB disk limit
            maxNetworkMB: 100 // 100MB network limit
        };
    }
    
    formatBytes(bytes) {
        if (bytes === 0) return '0 B';
        
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
    
    getFormattedStats() {
        const stats = this.getCurrentStats();
        
        return {
            cpu: {
                usage: `${stats.cpu.usage.toFixed(1)}%`,
                cores: stats.cpu.cores,
                loadAvg: stats.cpu.loadAvg?.map(load => load.toFixed(2)).join(', ')
            },
            memory: {
                used: this.formatBytes(stats.memory.used),
                total: this.formatBytes(stats.memory.total),
                available: this.formatBytes(stats.memory.available),
                usage: `${stats.memory.usagePercent.toFixed(1)}%`
            },
            disk: {
                read: this.formatBytes(stats.disk.readBytes) + '/s',
                write: this.formatBytes(stats.disk.writeBytes) + '/s'
            },
            network: {
                rx: this.formatBytes(stats.network.rx) + '/s',
                tx: this.formatBytes(stats.network.tx) + '/s'
            },
            timestamp: stats.timestamp.toISOString()
        };
    }
}

module.exports = SystemMonitor;
