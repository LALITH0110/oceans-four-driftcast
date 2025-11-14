// draw.js - Rendering functions for map and particles

import * as physics from './physics.js';

let projection;
let coastlineData = null;
let watercolorPattern = null;

/**
 * Initialize projection
 */
export function initProjection(width, height) {
    projection = d3.geoMercator()
        .center([-40, 32.5])
        .scale(width / 2.5)
        .translate([width / 2, height / 2]);

    return projection;
}

/**
 * Load coastline data
 */
export async function loadCoastline() {
    try {
        const response = await fetch('https://cdn.jsdelivr.net/npm/world-atlas@2/countries-110m.json');
        const data = await response.json();
        coastlineData = topojson.feature(data, data.objects.countries);
        return true;
    } catch (error) {
        console.warn('Failed to load TopoJSON, using fallback coastline');
        createFallbackCoastline();
        return false;
    }
}

/**
 * Create simple fallback coastline
 */
function createFallbackCoastline() {
    // Simple coastline approximation for North Atlantic
    coastlineData = {
        type: "FeatureCollection",
        features: [{
            type: "Feature",
            geometry: {
                type: "MultiLineString",
                coordinates: [
                    // US East Coast
                    [[-81, 25], [-80, 27], [-79, 32], [-76, 37], [-74, 40], [-71, 42], [-70, 43], [-68, 45], [-66, 47], [-62, 49]],
                    // Gulf of St. Lawrence
                    [[-62, 49], [-60, 50], [-58, 49], [-56, 48], [-54, 47]],
                    // Europe West Coast
                    [[-10, 36], [-9, 38], [-9, 41], [-8, 43], [-5, 44], [-2, 45], [0, 49], [1, 51], [2, 52], [4, 52], [6, 53], [8, 54], [10, 56], [11, 58]],
                    // Iceland outline
                    [[-24, 64], [-22, 66], [-16, 66], [-14, 65], [-14, 64], [-24, 64]],
                    // Mediterranean connection
                    [[0, 36], [5, 37], [10, 40], [15, 41], [20, 38]]
                ]
            }
        }]
    };
}

/**
 * Create watercolor texture
 */
export function createWatercolorTexture(ctx) {
    const canvas = document.createElement('canvas');
    canvas.width = 128;
    canvas.height = 128;
    const tempCtx = canvas.getContext('2d');

    // Create a noise-based watercolor effect
    const imageData = tempCtx.createImageData(128, 128);
    const data = imageData.data;

    for (let i = 0; i < 128; i++) {
        for (let j = 0; j < 128; j++) {
            const idx = (i * 128 + j) * 4;
            const noise = Math.random() * 0.15 + 0.85;

            // Ocean blue with variation
            data[idx] = 15 * noise;     // R
            data[idx + 1] = 50 * noise; // G
            data[idx + 2] = 80 * noise; // B
            data[idx + 3] = 255;         // A
        }
    }

    tempCtx.putImageData(imageData, 0, 0);
    watercolorPattern = ctx.createPattern(canvas, 'repeat');

    return watercolorPattern;
}

/**
 * Draw base map
 */
export function drawBaseMap(ctx, width, height, mode) {
    // Clear
    ctx.fillStyle = mode === 'heatmap' ? '#0d1b2a' : '#0a1a2e';
    ctx.fillRect(0, 0, width, height);

    // Draw ocean background
    if (mode === 'heatmap' && watercolorPattern) {
        ctx.fillStyle = watercolorPattern;
        ctx.fillRect(0, 0, width, height);
    }

    // Draw coastlines
    if (coastlineData && projection) {
        const path = d3.geoPath(projection, ctx);

        ctx.strokeStyle = mode === 'heatmap' ? 'rgba(100, 120, 140, 0.6)' : 'rgba(100, 150, 200, 0.8)';
        ctx.lineWidth = mode === 'heatmap' ? 0.5 : 1;
        ctx.beginPath();
        path(coastlineData);
        ctx.stroke();

        // Fill land
        ctx.fillStyle = mode === 'heatmap' ? 'rgba(40, 50, 60, 0.3)' : 'rgba(30, 40, 50, 0.5)';
        ctx.beginPath();
        path(coastlineData);
        ctx.fill();
    }

    // Draw graticule (grid lines)
    if (mode !== 'heatmap') {
        drawGraticule(ctx);
    }

    // Draw release sites if in pins mode
    if (mode === 'pins') {
        drawReleaseSites(ctx);
    }
}

/**
 * Draw graticule
 */
function drawGraticule(ctx) {
    if (!projection) return;

    ctx.strokeStyle = 'rgba(100, 150, 200, 0.15)';
    ctx.lineWidth = 0.5;

    // Longitude lines
    for (let lon = -100; lon <= 20; lon += 10) {
        ctx.beginPath();
        for (let lat = 0; lat <= 65; lat += 1) {
            const [x, y] = projection([lon, lat]);
            if (lat === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();
    }

    // Latitude lines
    for (let lat = 0; lat <= 65; lat += 10) {
        ctx.beginPath();
        for (let lon = -100; lon <= 20; lon += 1) {
            const [x, y] = projection([lon, lat]);
            if (lon === -100) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();
    }
}

/**
 * Draw release sites
 */
function drawReleaseSites(ctx) {
    if (!projection) return;

    const allSites = [...physics.RELEASE_US, ...physics.RELEASE_EU];

    for (const site of allSites) {
        const [x, y] = projection([site.lon, site.lat]);

        // Draw pin
        ctx.fillStyle = 'rgba(0, 212, 255, 0.8)';
        ctx.beginPath();
        ctx.arc(x, y, 3 + site.weight, 0, 2 * Math.PI);
        ctx.fill();

        // Draw pulse
        const pulseRadius = 5 + site.weight * 2 + Math.sin(Date.now() / 300) * 2;
        ctx.strokeStyle = 'rgba(0, 212, 255, 0.4)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(x, y, pulseRadius, 0, 2 * Math.PI);
        ctx.stroke();
    }
}

/**
 * Draw particles
 */
export function drawParticles(ctx, particles, mode, showTrails) {
    if (!projection) return;

    ctx.save();

    if (mode === 'currents') {
        // Draw as streaks
        for (let i = 0; i < particles.count; i++) {
            const idx = i * 8;
            if (particles.beached[i]) continue;

            const lon = particles.lon[idx];
            const lat = particles.lat[idx];
            const age = particles.age[i];

            const [x, y] = projection([lon, lat]);

            // Get velocity for streak direction
            const vel = physics.getVelocity(lon, lat, Date.now() / 1000, 'JJA', 1.0);
            const speed = Math.sqrt(vel.u ** 2 + vel.v ** 2);
            const angle = Math.atan2(vel.v, vel.u);

            const streakLength = Math.min(speed * 20, 15);

            const opacity = Math.min(age / 100, 0.6);
            const colorScale = chroma.scale(['#00d4ff', '#00ff88', '#ffaa00']).mode('lab');
            const color = colorScale(Math.min(speed / 0.5, 1)).alpha(opacity).css();

            ctx.strokeStyle = color;
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(x, y);
            ctx.lineTo(x - Math.cos(angle) * streakLength, y - Math.sin(angle) * streakLength);
            ctx.stroke();
        }
    } else if (mode === 'tracker') {
        // Draw thick neon trajectory
        if (particles.count > 0) {
            ctx.strokeStyle = '#00ff88';
            ctx.lineWidth = 3;
            ctx.shadowBlur = 8;
            ctx.shadowColor = '#00ff88';

            ctx.beginPath();
            let first = true;
            for (let i = 0; i < particles.count; i++) {
                const idx = i * 8;
                const lon = particles.lon[idx];
                const lat = particles.lat[idx];
                const [x, y] = projection([lon, lat]);

                if (first) {
                    ctx.moveTo(x, y);
                    first = false;
                } else {
                    ctx.lineTo(x, y);
                }
            }
            ctx.stroke();

            // Draw current position
            const lastIdx = (particles.count - 1) * 8;
            const [lastX, lastY] = projection([particles.lon[lastIdx], particles.lat[lastIdx]]);

            ctx.fillStyle = '#00ff88';
            ctx.shadowBlur = 15;
            ctx.beginPath();
            ctx.arc(lastX, lastY, 5, 0, 2 * Math.PI);
            ctx.fill();
        }
    } else {
        // Regular particle rendering
        for (let i = 0; i < particles.count; i++) {
            const idx = i * 8;
            if (particles.beached[i]) continue;

            const lon = particles.lon[idx];
            const lat = particles.lat[idx];
            const age = particles.age[i];

            const [x, y] = projection([lon, lat]);

            const opacity = Math.min(age / 50, 0.8);
            const size = 1.5;

            ctx.fillStyle = `rgba(0, 212, 255, ${opacity})`;
            ctx.beginPath();
            ctx.arc(x, y, size, 0, 2 * Math.PI);
            ctx.fill();

            // Draw trail
            if (showTrails && particles.trail && particles.trail[i]) {
                ctx.strokeStyle = `rgba(0, 212, 255, 0.2)`;
                ctx.lineWidth = 0.5;
                ctx.beginPath();

                const trail = particles.trail[i];
                for (let j = 0; j < trail.length; j++) {
                    const [tx, ty] = projection([trail[j].lon, trail[j].lat]);
                    if (j === 0) ctx.moveTo(tx, ty);
                    else ctx.lineTo(tx, ty);
                }
                ctx.stroke();
            }
        }
    }

    ctx.restore();
}

/**
 * Draw scalar field overlay (for currents mode)
 */
export function drawScalarOverlay(ctx, width, height, time, season, windageMultiplier) {
    if (!projection) return;

    const resolution = 20; // Grid resolution
    const cols = Math.floor(width / resolution);
    const rows = Math.floor(height / resolution);

    const colorScale = chroma.scale(['#001a33', '#004d7a', '#008793', '#00bf72']).mode('lab');

    for (let i = 0; i < cols; i++) {
        for (let j = 0; j < rows; j++) {
            const x = i * resolution;
            const y = j * resolution;

            const [lon, lat] = projection.invert([x, y]);

            if (lon < -100 || lon > 20 || lat < 0 || lat > 65) continue;

            const speed = physics.getSpeed(lon, lat, time, season, windageMultiplier);
            const normalizedSpeed = Math.min(speed / 0.5, 1);

            const color = colorScale(normalizedSpeed).alpha(0.15).css();

            ctx.fillStyle = color;
            ctx.fillRect(x, y, resolution, resolution);
        }
    }
}

/**
 * Draw heatmap
 */
export function drawHeatmap(ctx, width, height, particles, simulationYears) {
    if (!projection) return;

    const resolution = 10;
    const cols = Math.floor(width / resolution);
    const rows = Math.floor(height / resolution);

    // Create density grid
    const grid = new Float32Array(cols * rows);

    for (let i = 0; i < particles.count; i++) {
        const idx = i * 8;
        const lon = particles.lon[idx];
        const lat = particles.lat[idx];

        const [x, y] = projection([lon, lat]);
        const col = Math.floor(x / resolution);
        const row = Math.floor(y / resolution);

        if (col >= 0 && col < cols && row >= 0 && row < rows) {
            grid[row * cols + col]++;
        }
    }

    // Find max for normalization
    let maxDensity = 0;
    for (let i = 0; i < grid.length; i++) {
        maxDensity = Math.max(maxDensity, grid[i]);
    }

    // Draw heatmap
    const colorScale = chroma.scale(['rgba(0,255,0,0)', 'rgba(0,255,0,0.3)', 'rgba(255,255,0,0.6)', 'rgba(255,100,0,0.8)', 'rgba(255,0,0,0.9)']).mode('lab');

    for (let i = 0; i < cols; i++) {
        for (let j = 0; j < rows; j++) {
            const density = grid[j * cols + i];
            if (density === 0) continue;

            const normalized = density / maxDensity;
            const color = colorScale(normalized).css();

            ctx.fillStyle = color;
            ctx.fillRect(i * resolution, j * resolution, resolution, resolution);
        }
    }

    // Draw caption
    ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
    ctx.font = '14px monospace';
    ctx.fillText(`After ${simulationYears.toFixed(1)} synthetic years`, 20, height - 20);
}

/**
 * Update legend
 */
export function updateLegend(mode, season) {
    const legend = document.getElementById('legend');

    if (mode === 'currents') {
        legend.innerHTML = `
            <h4>Current Speed</h4>
            <div class="legend-scale" id="legend-scale"></div>
            <div class="legend-labels">
                <span>0 m/s</span>
                <span>0.5 m/s</span>
            </div>
        `;

        // Fill scale gradient
        const scale = document.getElementById('legend-scale');
        const colorScale = chroma.scale(['#00d4ff', '#00ff88', '#ffaa00']).mode('lab');
        const gradient = [];
        for (let i = 0; i <= 10; i++) {
            gradient.push(colorScale(i / 10).css());
        }
        scale.style.background = `linear-gradient(to right, ${gradient.join(', ')})`;

        legend.style.display = 'block';
    } else {
        legend.style.display = 'none';
    }
}

/**
 * Get projection (for external use)
 */
export function getProjection() {
    return projection;
}
