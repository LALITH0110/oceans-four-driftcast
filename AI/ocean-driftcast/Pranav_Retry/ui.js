// ui.js - UI controls and interactions

import * as physics from './physics.js';
import * as draw from './draw.js';

let app = null;
let mediaRecorder = null;
let recordedChunks = [];
let recordingStartTime = 0;
let recordingDuration = 300000; // 5 minutes in ms

/**
 * Initialize UI
 */
export function initUI(appInstance) {
    app = appInstance;

    // Mode selection
    document.getElementById('mode-select').addEventListener('change', (e) => {
        app.mode = e.target.value;
        if (app.mode === 'tracker') {
            // Reset particles for tracker mode
            app.resetParticles();
        }
        draw.updateLegend(app.mode, app.season);
        app.needsRedraw = true;
    });

    // Playback controls
    document.getElementById('btn-start').addEventListener('click', () => {
        app.running = true;
    });

    document.getElementById('btn-pause').addEventListener('click', () => {
        app.running = false;
    });

    document.getElementById('btn-reset').addEventListener('click', () => {
        app.resetParticles();
        app.beachedCount = 0;
        app.simulationTime = 0;
        updateStats();
    });

    // Quick release
    document.getElementById('btn-drop-us').addEventListener('click', () => {
        releaseFromSites(physics.RELEASE_US);
    });

    document.getElementById('btn-drop-eu').addEventListener('click', () => {
        releaseFromSites(physics.RELEASE_EU);
    });

    // Toggles
    document.getElementById('toggle-trails').addEventListener('change', (e) => {
        app.showTrails = e.target.checked;
        app.needsRedraw = true;
    });

    document.getElementById('toggle-heatmap').addEventListener('change', (e) => {
        app.showHeatmap = e.target.checked;
        app.needsRedraw = true;
    });

    document.getElementById('toggle-beaching').addEventListener('change', (e) => {
        app.beachingEnabled = e.target.checked;
    });

    // Sliders
    document.getElementById('slider-particles').addEventListener('input', (e) => {
        app.maxParticles = parseInt(e.target.value);
        document.getElementById('particles-val').textContent = app.maxParticles;
    });

    document.getElementById('slider-speed').addEventListener('input', (e) => {
        app.speedMultiplier = parseFloat(e.target.value);
        document.getElementById('speed-val').textContent = app.speedMultiplier.toFixed(1);
    });

    document.getElementById('slider-windage').addEventListener('input', (e) => {
        app.windageMultiplier = parseFloat(e.target.value);
        document.getElementById('windage-val').textContent = app.windageMultiplier.toFixed(1);
    });

    // Season
    document.getElementById('season-select').addEventListener('change', (e) => {
        app.season = e.target.value;
        draw.updateLegend(app.mode, app.season);
    });

    // Export
    document.getElementById('btn-export').addEventListener('click', startRecording);
    document.getElementById('btn-stop-export').addEventListener('click', stopRecording);

    // Canvas interactions
    const baseCanvas = document.getElementById('base-canvas');
    baseCanvas.addEventListener('mousemove', handleMouseMove);
    baseCanvas.addEventListener('click', handleClick);
    baseCanvas.addEventListener('mouseleave', hideCursorInfo);

    // Release site filters
    document.getElementById('show-us').addEventListener('change', updateReleaseList);
    document.getElementById('show-eu').addEventListener('change', updateReleaseList);

    // Initialize release list
    updateReleaseList();
}

/**
 * Update release site list
 */
function updateReleaseList() {
    const listEl = document.getElementById('release-list');
    const showUS = document.getElementById('show-us').checked;
    const showEU = document.getElementById('show-eu').checked;

    listEl.innerHTML = '';

    const sites = [];
    if (showUS) sites.push(...physics.RELEASE_US.map(s => ({...s, region: 'US'})));
    if (showEU) sites.push(...physics.RELEASE_EU.map(s => ({...s, region: 'EU'})));

    for (const site of sites) {
        const item = document.createElement('div');
        item.className = 'release-item';
        item.innerHTML = `
            <div class="name">${site.name}</div>
            <div class="coords">${site.lat.toFixed(2)}°N, ${Math.abs(site.lon).toFixed(2)}°${site.lon < 0 ? 'W' : 'E'}</div>
            <div class="weight">${site.weight}x</div>
        `;

        item.addEventListener('click', () => {
            releaseFromSite(site);
            item.classList.add('active');
            setTimeout(() => item.classList.remove('active'), 1000);
        });

        listEl.appendChild(item);
    }
}

/**
 * Release particles from a single site
 */
function releaseFromSite(site) {
    if (!app) return;

    const count = Math.floor(100 * site.weight);

    for (let i = 0; i < count; i++) {
        // Add random offset
        const offset = 0.5;
        const lon = site.lon + (Math.random() - 0.5) * offset;
        const lat = site.lat + (Math.random() - 0.5) * offset;

        app.addParticle(lon, lat);
    }
}

/**
 * Release particles from multiple sites
 */
function releaseFromSites(sites) {
    for (const site of sites) {
        releaseFromSite(site);
    }
}

/**
 * Handle mouse move on canvas
 */
function handleMouseMove(event) {
    if (!app || app.mode !== 'currents') {
        hideCursorInfo();
        return;
    }

    const rect = event.target.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    const projection = draw.getProjection();
    if (!projection) return;

    const [lon, lat] = projection.invert([x, y]);

    // Check bounds
    if (lon < -100 || lon > 20 || lat < 0 || lat > 65) {
        hideCursorInfo();
        return;
    }

    // Get current info
    const speed = physics.getSpeed(lon, lat, Date.now() / 1000, app.season, app.windageMultiplier);
    const direction = physics.getDirection(lon, lat, Date.now() / 1000, app.season, app.windageMultiplier);

    // Show info card
    const infoCard = document.getElementById('cursor-info');
    infoCard.classList.add('visible');
    infoCard.style.left = (x + 15) + 'px';
    infoCard.style.top = (y + 15) + 'px';

    const directionLabel = getDirectionLabel(direction);

    infoCard.innerHTML = `
        <div><strong>Position</strong></div>
        <div>Lat: ${lat.toFixed(2)}°N</div>
        <div>Lon: ${Math.abs(lon).toFixed(2)}°${lon < 0 ? 'W' : 'E'}</div>
        <div><strong>Current</strong></div>
        <div>Speed: ${(speed * 100).toFixed(1)} cm/s</div>
        <div>Direction: ${directionLabel} (${direction.toFixed(0)}°)</div>
    `;
}

/**
 * Hide cursor info
 */
function hideCursorInfo() {
    const infoCard = document.getElementById('cursor-info');
    infoCard.classList.remove('visible');
}

/**
 * Get direction label
 */
function getDirectionLabel(degrees) {
    const directions = ['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE'];
    const index = Math.round(degrees / 45) % 8;
    return directions[index];
}

/**
 * Handle canvas click
 */
function handleClick(event) {
    if (!app) return;

    const rect = event.target.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    const projection = draw.getProjection();
    if (!projection) return;

    const [lon, lat] = projection.invert([x, y]);

    if (app.mode === 'pins') {
        // Release particles at click location
        for (let i = 0; i < 50; i++) {
            const offset = 0.3;
            const pLon = lon + (Math.random() - 0.5) * offset;
            const pLat = lat + (Math.random() - 0.5) * offset;
            app.addParticle(pLon, pLat);
        }
    } else if (app.mode === 'tracker') {
        // Start new tracker
        app.resetParticles();
        app.addParticle(lon, lat);
        app.trackerStartLon = lon;
        app.trackerStartLat = lat;
        app.trackerDistance = 0;
    }
}

/**
 * Update stats display
 */
export function updateStats() {
    if (!app) return;

    document.getElementById('fps').textContent = app.fps.toFixed(0);
    document.getElementById('beached-count').textContent = app.beachedCount;
    document.getElementById('active-count').textContent = app.particles.count;
}

/**
 * Start WebM recording
 */
async function startRecording() {
    if (!app) return;

    const canvas = document.getElementById('anim-canvas');

    try {
        // Create a stream from both canvases
        const baseCanvas = document.getElementById('base-canvas');
        const animCanvas = document.getElementById('anim-canvas');

        // Create a temporary canvas to combine both
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = 1920;
        tempCanvas.height = 1080;
        const tempCtx = tempCanvas.getContext('2d');

        // Set up MediaRecorder
        const stream = tempCanvas.captureStream(30);
        mediaRecorder = new MediaRecorder(stream, {
            mimeType: 'video/webm;codecs=vp9',
            videoBitsPerSecond: 5000000
        });

        recordedChunks = [];

        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                recordedChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = () => {
            const blob = new Blob(recordedChunks, { type: 'video/webm' });
            const url = URL.createObjectURL(blob);

            const a = document.createElement('a');
            a.href = url;
            a.download = `driftcast-${Date.now()}.webm`;
            a.click();

            URL.revokeObjectURL(url);

            document.getElementById('btn-export').style.display = 'block';
            document.getElementById('btn-stop-export').style.display = 'none';
            document.getElementById('export-progress').innerHTML = '';
        };

        // Start recording
        mediaRecorder.start();
        recordingStartTime = Date.now();

        document.getElementById('btn-export').style.display = 'none';
        document.getElementById('btn-stop-export').style.display = 'block';

        // Update progress
        const updateProgress = () => {
            if (!mediaRecorder || mediaRecorder.state !== 'recording') return;

            const elapsed = Date.now() - recordingStartTime;
            const progress = Math.min((elapsed / recordingDuration) * 100, 100);

            document.getElementById('export-progress').innerHTML = `
                <div>Recording: ${(elapsed / 1000).toFixed(0)}s / ${(recordingDuration / 1000).toFixed(0)}s</div>
                <div class="progress-bar"><div class="progress-fill" style="width: ${progress}%"></div></div>
            `;

            // Copy canvases to temp canvas
            tempCtx.drawImage(baseCanvas, 0, 0, tempCanvas.width, tempCanvas.height);
            tempCtx.drawImage(animCanvas, 0, 0, tempCanvas.width, tempCanvas.height);

            // Auto-stop after duration
            if (elapsed >= recordingDuration) {
                stopRecording();
                return;
            }

            requestAnimationFrame(updateProgress);
        };

        updateProgress();

        console.log('Recording started');
    } catch (error) {
        console.error('Failed to start recording:', error);
        alert('Failed to start recording. Your browser may not support WebM recording.');
    }
}

/**
 * Stop WebM recording
 */
function stopRecording() {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
        console.log('Recording stopped');
    }
}

/**
 * Update tracker info
 */
export function updateTrackerInfo(distance, probability) {
    if (!app || app.mode !== 'tracker') return;

    const legend = document.getElementById('legend');
    legend.innerHTML = `
        <h4>Tracker</h4>
        <div>Distance: ${distance.toFixed(0)} km</div>
        <div>Open Ocean: ${(probability * 100).toFixed(0)}%</div>
    `;
    legend.style.display = 'block';
}
