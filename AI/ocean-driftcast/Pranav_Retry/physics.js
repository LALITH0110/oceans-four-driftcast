// physics.js - Ocean current simulation and particle dynamics

// Release sites
export const RELEASE_US = [
  {name:"Miami/Florida Straits", lat:25.77, lon:-80.19, weight:1.2},
  {name:"Jacksonville", lat:30.33, lon:-81.65, weight:0.9},
  {name:"Savannah", lat:32.08, lon:-81.09, weight:0.8},
  {name:"Charleston", lat:32.78, lon:-79.93, weight:0.8},
  {name:"Cape Hatteras", lat:35.26, lon:-75.54, weight:1.1},
  {name:"Chesapeake Bay", lat:37.00, lon:-76.30, weight:1.3},
  {name:"Delaware Bay", lat:39.00, lon:-75.00, weight:0.9},
  {name:"NY/NJ Harbor", lat:40.67, lon:-74.02, weight:1.6},
  {name:"Narragansett", lat:41.49, lon:-71.32, weight:0.7},
  {name:"Massachusetts Bay", lat:42.40, lon:-70.70, weight:0.9},
  {name:"Gulf of St. Lawrence", lat:48.00, lon:-61.00, weight:0.7},
  {name:"San Juan, PR", lat:18.47, lon:-66.11, weight:0.6}
];

export const RELEASE_EU = [
  {name:"Thames Estuary", lat:51.50, lon:0.00, weight:1.2},
  {name:"Seine Estuary", lat:49.49, lon:0.12, weight:1.0},
  {name:"Loire Estuary", lat:47.20, lon:-2.17, weight:0.8},
  {name:"Gironde/Bordeaux", lat:45.55, lon:-1.15, weight:1.0},
  {name:"Bay of Biscay", lat:43.60, lon:-1.50, weight:1.4},
  {name:"Tagus/Lisbon", lat:38.70, lon:-9.15, weight:1.1},
  {name:"Douro/Porto", lat:41.15, lon:-8.67, weight:0.9},
  {name:"Galicia", lat:42.90, lon:-9.27, weight:0.9},
  {name:"Rhine/Rotterdam", lat:51.95, lon:4.13, weight:1.2},
  {name:"Elbe/Hamburg", lat:53.55, lon:9.99, weight:0.8},
  {name:"Irish Sea", lat:53.80, lon:-3.50, weight:0.9},
  {name:"Skagerrak", lat:58.00, lon:9.00, weight:0.7}
];

// Coastal regions for beaching (simplified polygons)
const COASTAL_REGIONS = [
  {name: "Mid-Atlantic US", points: [{lat:32, lon:-81}, {lat:42, lon:-74}, {lat:42, lon:-69}, {lat:32, lon:-76}], baseProbability: 0.15},
  {name: "Bay of Biscay", points: [{lat:43, lon:-5}, {lat:48, lon:-5}, {lat:48, lon:0}, {lat:43, lon:0}], baseProbability: 0.18},
  {name: "West Iberia", points: [{lat:37, lon:-10}, {lat:43, lon:-10}, {lat:43, lon:-8}, {lat:37, lon:-8}], baseProbability: 0.16},
  {name: "French Atlantic", points: [{lat:46, lon:-4}, {lat:49, lon:-4}, {lat:49, lon:-1}, {lat:46, lon:-1}], baseProbability: 0.17},
  {name: "Irish Sea", points: [{lat:52, lon:-6}, {lat:55, lon:-6}, {lat:55, lon:-3}, {lat:52, lon:-3}], baseProbability: 0.14},
  {name: "Skagerrak", points: [{lat:57, lon:8}, {lat:59, lon:8}, {lat:59, lon:12}, {lat:57, lon:12}], baseProbability: 0.13}
];

// Season windage multipliers
const SEASON_WINDAGE = {
  DJF: 1.4,  // Winter - strongest
  MAM: 1.1,  // Spring
  JJA: 0.7,  // Summer - weakest
  SON: 1.0   // Fall
};

// Season beaching multipliers
const SEASON_BEACHING = {
  DJF: 2.0,
  MAM: 1.5,
  JJA: 0.7,
  SON: 1.2
};

// Physics constants
const EARTH_RADIUS_KM = 6371.0;
const DEG_TO_RAD = Math.PI / 180;
const TIME_STEP_HOURS = 4; // 4 hours per frame
const TIME_STEP_DAYS = TIME_STEP_HOURS / 24;

// Gyre parameters
const GYRE_CENTER_LAT = 30;
const GYRE_CENTER_LON = -60;
const GYRE_SIGMA_LAT = 12;
const GYRE_SIGMA_LON = 20;
const GYRE_STRENGTH = 0.25;

// Gulf Stream parameters
const GS_LAT = 37;
const GS_WIDTH = 3;
const GS_STRENGTH = 0.8;
const GS_MEANDER_AMP = 2;
const GS_MEANDER_WAVELENGTH = 15;

// Noise parameters
let noiseTime = 0;

/**
 * Compute ocean current velocity at a given position and time
 */
export function getVelocity(lon, lat, time, season, windageMultiplier) {
    let u = 0, v = 0;

    // 1. Clockwise gyre using Gaussian streamfunction
    const dx = lon - GYRE_CENTER_LON;
    const dy = lat - GYRE_CENTER_LAT;
    const exp_term = Math.exp(-(dx*dx)/(2*GYRE_SIGMA_LON*GYRE_SIGMA_LON) - (dy*dy)/(2*GYRE_SIGMA_LAT*GYRE_SIGMA_LAT));

    // u = ∂ψ/∂y, v = -∂ψ/∂x (for clockwise rotation)
    const psi = GYRE_STRENGTH * exp_term;
    u += psi * (-dy / (GYRE_SIGMA_LAT * GYRE_SIGMA_LAT));
    v += psi * (dx / (GYRE_SIGMA_LON * GYRE_SIGMA_LON));

    // 2. Gulf Stream jet (eastward with meander)
    const gs_dist = lat - GS_LAT - GS_MEANDER_AMP * Math.sin(2 * Math.PI * lon / GS_MEANDER_WAVELENGTH);
    if (Math.abs(gs_dist) < GS_WIDTH * 2) {
        const gs_profile = GS_STRENGTH * Math.exp(-(gs_dist * gs_dist) / (GS_WIDTH * GS_WIDTH));
        u += gs_profile; // Eastward flow
    }

    // 3. Azores Current (eastward around 35N)
    if (lat > 33 && lat < 38 && lon > -40 && lon < -10) {
        const azores_strength = 0.15 * Math.exp(-((lat - 35.5) * (lat - 35.5)) / 4);
        u += azores_strength;
    }

    // 4. Canary Current (southward along Iberia)
    if (lat > 25 && lat < 43 && lon > -18 && lon < -8) {
        const canary_strength = 0.2 * Math.exp(-((lon + 13) * (lon + 13)) / 9);
        v -= canary_strength; // Southward
    }

    // 5. Windage component (seasonal)
    const seasonalWindage = SEASON_WINDAGE[season] || 1.0;
    const windage = 0.03 * seasonalWindage * windageMultiplier;

    // Predominant westerly winds in North Atlantic
    u += windage * 0.6;
    v += windage * 0.2 * Math.sin(lat * DEG_TO_RAD * 2);

    // 6. Time-varying noise
    const noise_u = 0.02 * Math.sin(time * 0.1 + lon * 0.5 + lat * 0.3);
    const noise_v = 0.02 * Math.cos(time * 0.1 + lon * 0.4 + lat * 0.6);
    u += noise_u;
    v += noise_v;

    return {u, v};
}

/**
 * RK4 integrator for particle position
 */
export function integrateRK4(lon, lat, time, season, windageMultiplier, speedMultiplier) {
    const dt = TIME_STEP_DAYS * speedMultiplier;

    // k1
    const v1 = getVelocity(lon, lat, time, season, windageMultiplier);
    const k1_lon = v1.u * dt;
    const k1_lat = v1.v * dt;

    // k2
    const v2 = getVelocity(lon + k1_lon/2, lat + k1_lat/2, time + dt/2, season, windageMultiplier);
    const k2_lon = v2.u * dt;
    const k2_lat = v2.v * dt;

    // k3
    const v3 = getVelocity(lon + k2_lon/2, lat + k2_lat/2, time + dt/2, season, windageMultiplier);
    const k3_lon = v3.u * dt;
    const k3_lat = v3.v * dt;

    // k4
    const v4 = getVelocity(lon + k3_lon, lat + k3_lat, time + dt, season, windageMultiplier);
    const k4_lon = v4.u * dt;
    const k4_lat = v4.v * dt;

    // Combine
    const new_lon = lon + (k1_lon + 2*k2_lon + 2*k3_lon + k4_lon) / 6;
    const new_lat = lat + (k1_lat + 2*k2_lat + 2*k3_lat + k4_lat) / 6;

    return {lon: new_lon, lat: new_lat};
}

/**
 * Check if point is inside a polygon (ray casting algorithm)
 */
function pointInPolygon(lat, lon, polygon) {
    let inside = false;
    for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
        const xi = polygon[i].lon, yi = polygon[i].lat;
        const xj = polygon[j].lon, yj = polygon[j].lat;

        const intersect = ((yi > lat) !== (yj > lat))
            && (lon < (xj - xi) * (lat - yi) / (yj - yi) + xi);
        if (intersect) inside = !inside;
    }
    return inside;
}

/**
 * Get distance to nearest coast (simplified)
 */
function getDistanceToCoast(lon, lat) {
    // Simple heuristic based on known coastal positions
    const coastalPoints = [
        {lat: 40, lon: -74}, // US East
        {lat: 35, lon: -76},
        {lat: 45, lon: -1},  // France
        {lat: 40, lon: -9},  // Portugal
        {lat: 53, lon: -4},  // Ireland/UK
        {lat: 58, lon: 10}   // Scandinavia
    ];

    let minDist = Infinity;
    for (const point of coastalPoints) {
        const dist = Math.sqrt((lat - point.lat)**2 + (lon - point.lon)**2);
        minDist = Math.min(minDist, dist);
    }

    return minDist * 111; // Convert to km (approximate)
}

/**
 * Check if particle should beach
 */
export function checkBeaching(lon, lat, velocity, season) {
    // Check if in any coastal region
    for (const region of COASTAL_REGIONS) {
        if (pointInPolygon(lat, lon, region.points)) {
            const distance = getDistanceToCoast(lon, lat);

            // If close to coast (< 50km) and moving onshore
            if (distance < 50) {
                // Check if velocity has onshore component
                const speed = Math.sqrt(velocity.u ** 2 + velocity.v ** 2);

                // Compute rough onshore direction (toward nearest coast)
                // Simplified: check if moving toward center of region
                const regionCenterLat = region.points.reduce((sum, p) => sum + p.lat, 0) / region.points.length;
                const regionCenterLon = region.points.reduce((sum, p) => sum + p.lon, 0) / region.points.length;

                const towardCoast = (velocity.u * (regionCenterLon - lon) + velocity.v * (regionCenterLat - lat)) > 0;

                if (towardCoast) {
                    // Calculate beaching probability
                    const seasonMultiplier = SEASON_BEACHING[season] || 1.0;
                    const probability = region.baseProbability * seasonMultiplier * (50 - distance) / 50;

                    if (Math.random() < probability) {
                        return {beached: true, region: region.name};
                    }
                }
            }
        }
    }

    return {beached: false};
}

/**
 * Calculate distance between two lat/lon points (Haversine formula)
 */
export function calculateDistance(lat1, lon1, lat2, lon2) {
    const dLat = (lat2 - lat1) * DEG_TO_RAD;
    const dLon = (lon2 - lon1) * DEG_TO_RAD;

    const a = Math.sin(dLat/2) * Math.sin(dLat/2) +
              Math.cos(lat1 * DEG_TO_RAD) * Math.cos(lat2 * DEG_TO_RAD) *
              Math.sin(dLon/2) * Math.sin(dLon/2);

    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
    return EARTH_RADIUS_KM * c;
}

/**
 * Calculate "open ocean probability" for tracker mode
 * Based on distance from coast and current patterns
 */
export function calculateOpenOceanProbability(lon, lat) {
    const distToCoast = getDistanceToCoast(lon, lat);

    // Probability increases with distance from coast
    let probability = Math.min(distToCoast / 500, 1.0); // Max at 500km

    // Increase probability if in gyre center
    const dx = lon - GYRE_CENTER_LON;
    const dy = lat - GYRE_CENTER_LAT;
    const gyreDistance = Math.sqrt(dx*dx + dy*dy);
    if (gyreDistance < 15) {
        probability += 0.2;
    }

    return Math.min(probability, 1.0);
}

/**
 * Get speed magnitude at position
 */
export function getSpeed(lon, lat, time, season, windageMultiplier) {
    const vel = getVelocity(lon, lat, time, season, windageMultiplier);
    return Math.sqrt(vel.u * vel.u + vel.v * vel.v);
}

/**
 * Get direction at position (in degrees, 0 = East, 90 = North)
 */
export function getDirection(lon, lat, time, season, windageMultiplier) {
    const vel = getVelocity(lon, lat, time, season, windageMultiplier);
    let angle = Math.atan2(vel.v, vel.u) * 180 / Math.PI;
    return (angle + 360) % 360;
}
