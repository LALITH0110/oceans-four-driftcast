// physics.js
// Synthetic North Atlantic flow model and particle integrator

export const DOMAIN = Object.freeze({
  lonMin: -100,
  lonMax: 20,
  latMin: 0,
  latMax: 65,
});

export const SEASONS = Object.freeze({
  DJF: { wind: 1.4, gust: 1.1, beach: 2.0 },
  MAM: { wind: 1.1, gust: 0.9, beach: 1.5 },
  JJA: { wind: 0.6, gust: 0.6, beach: 0.7 },
  SON: { wind: 1.0, gust: 0.8, beach: 1.2 },
});

export const RELEASE_US = Object.freeze([
  { name: "Miami/Florida Straits", lat: 25.77, lon: -80.19, weight: 1.2 },
  { name: "Jacksonville", lat: 30.33, lon: -81.65, weight: 0.9 },
  { name: "Savannah", lat: 32.08, lon: -81.09, weight: 0.8 },
  { name: "Charleston", lat: 32.78, lon: -79.93, weight: 0.8 },
  { name: "Cape Hatteras", lat: 35.26, lon: -75.54, weight: 1.1 },
  { name: "Chesapeake Bay", lat: 37.0, lon: -76.3, weight: 1.3 },
  { name: "Delaware Bay", lat: 39.0, lon: -75.0, weight: 0.9 },
  { name: "NY/NJ Harbor", lat: 40.67, lon: -74.02, weight: 1.6 },
  { name: "Narragansett", lat: 41.49, lon: -71.32, weight: 0.7 },
  { name: "Massachusetts Bay", lat: 42.4, lon: -70.7, weight: 0.9 },
  { name: "Gulf of St. Lawrence", lat: 48.0, lon: -61.0, weight: 0.7 },
  { name: "San Juan, PR", lat: 18.47, lon: -66.11, weight: 0.6 },
]);

export const RELEASE_EU = Object.freeze([
  { name: "Thames Estuary", lat: 51.5, lon: 0.0, weight: 1.2 },
  { name: "Seine Estuary", lat: 49.49, lon: 0.12, weight: 1.0 },
  { name: "Loire Estuary", lat: 47.2, lon: -2.17, weight: 0.8 },
  { name: "Gironde/Bordeaux", lat: 45.55, lon: -1.15, weight: 1.0 },
  { name: "Bay of Biscay", lat: 43.6, lon: -1.5, weight: 1.4 },
  { name: "Tagus/Lisbon", lat: 38.7, lon: -9.15, weight: 1.1 },
  { name: "Douro/Porto", lat: 41.15, lon: -8.67, weight: 0.9 },
  { name: "Galicia", lat: 42.9, lon: -9.27, weight: 0.9 },
  { name: "Rhine/Rotterdam", lat: 51.95, lon: 4.13, weight: 1.2 },
  { name: "Elbe/Hamburg", lat: 53.55, lon: 9.99, weight: 0.8 },
  { name: "Irish Sea", lat: 53.8, lon: -3.5, weight: 0.9 },
  { name: "Skagerrak", lat: 58.0, lon: 9.0, weight: 0.7 },
]);

const COAST_REGIONS = [
  { name: "US Mid-Atlantic", region: "us", lon0: -82, lon1: -66, lat0: 24, lat1: 45, normal: [+1, 0] },
  { name: "Newfoundland", region: "us", lon0: -62, lon1: -50, lat0: 45, lat1: 55, normal: [+1, 0] },
  { name: "Bay of Biscay", region: "eu", lon0: -10, lon1: 0, lat0: 43, lat1: 48, normal: [-1, 0] },
  { name: "West Iberia", region: "eu", lon0: -11, lon1: -6, lat0: 36, lat1: 43, normal: [-1, 0] },
  { name: "French Atlantic", region: "eu", lon0: -5, lon1: 3, lat0: 45, lat1: 50, normal: [-1, 0] },
  { name: "Irish Sea", region: "eu", lon0: -7.5, lon1: -3, lat0: 51, lat1: 55.5, normal: [-1, 0] },
  { name: "Skagerrak", region: "eu", lon0: 6, lon1: 12, lat0: 56, lat1: 59, normal: [-1, 0] },
];

export const METERS_PER_DEG_LAT = 111132; // average

export function metersPerDegLon(latDeg) {
  return 111320 * Math.cos((latDeg * Math.PI) / 180);
}

export function clamp(v, lo, hi) {
  return v < lo ? lo : v > hi ? hi : v;
}

export function wrapToDomain(lon, lat) {
  return [
    clamp(lon, DOMAIN.lonMin, DOMAIN.lonMax),
    clamp(lat, DOMAIN.latMin, DOMAIN.latMax),
  ];
}

export function createMulberry32(seed) {
  let t = seed >>> 0;
  return function mulberry() {
    t |= 0;
    t = (t + 0x6d2b79f5) | 0;
    let r = Math.imul(t ^ (t >>> 15), t | 1);
    r ^= r + Math.imul(r ^ (r >>> 7), r | 61);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

export function createParticlePool(maxParticles) {
  return {
    max: maxParticles,
    count: 0,
    lon: new Float32Array(maxParticles),
    lat: new Float32Array(maxParticles),
    u: new Float32Array(maxParticles),
    v: new Float32Array(maxParticles),
    scalar: new Float32Array(maxParticles),
    ageHours: new Float32Array(maxParticles),
    beached: new Uint8Array(maxParticles),
    regionTag: new Uint8Array(maxParticles), // 0=US,1=EU
  };
}

export function clearParticlePool(pool) {
  pool.count = 0;
  pool.beached.fill(0);
  pool.ageHours.fill(0);
}

function chooseRelease(set, rng) {
  const totalWeight = set.reduce((acc, site) => acc + site.weight, 0);
  const pick = rng() * totalWeight;
  let accum = 0;
  for (const site of set) {
    accum += site.weight;
    if (pick <= accum) return site;
  }
  return set[set.length - 1];
}

export function spawnParticles(pool, howMany, which, rng, jitter = 0.25) {
  const set = which === "EU" ? RELEASE_EU : RELEASE_US;
  const regionTag = which === "EU" ? 1 : 0;
  let inserted = 0;
  while (inserted < howMany && pool.count < pool.max) {
    const site = chooseRelease(set, rng);
    const idx = pool.count++;
    pool.lon[idx] = site.lon + (rng() * 2 - 1) * jitter;
    pool.lat[idx] = site.lat + (rng() * 2 - 1) * jitter;
    const [lonClamped, latClamped] = wrapToDomain(pool.lon[idx], pool.lat[idx]);
    pool.lon[idx] = lonClamped;
    pool.lat[idx] = latClamped;
    pool.u[idx] = 0;
    pool.v[idx] = 0;
    pool.ageHours[idx] = 0;
    pool.beached[idx] = 0;
    pool.regionTag[idx] = regionTag;
    inserted++;
  }
  return inserted;
}

export function velocityField(lon, lat, days, options) {
  const { season = "DJF", windage = 0.02, speedFactor = 1.0 } = options;
  const seasonal = SEASONS[season] || SEASONS.DJF;

  // Subtropical gyre streamfunction derivative (clockwise vortex)
  const dx = lon + 60;
  const dy = lat - 30;
  const r2 = (dx * dx + dy * dy) / (26 * 26);
  const amp = Math.exp(-r2);
  const uGyre = (2 * amp * dy) / (26 * 26);
  const vGyre = (-2 * amp * dx) / (26 * 26);

  // Gulf Stream jet
  const jetAxis = 37 + Math.sin(((lon + 80) / 18) * Math.PI * 2) * 1.6;
  const jetLatSpread = Math.exp(-((lat - jetAxis) ** 2) / (3.5 * 3.5));
  const jetLonTaper = Math.exp(-((lon + 70) ** 2) / (12 * 12));
  const uJet = 4.5 * jetLatSpread * jetLonTaper;
  const vJet = 0.35 * (Math.sin(((lat - 28) / 6) * Math.PI) * jetLonTaper);

  // Azores + Canary currents
  const uAzores =
    1.8 *
    Math.exp(-((lat - 32) ** 2) / (6.5 * 6.5)) *
    Math.exp(-((lon + 38) ** 2) / (28 * 28));
  const vCanary =
    -1.1 *
    Math.exp(-((lon + 10) ** 2) / (6 * 6)) *
    Math.exp(-((lat - 31) ** 2) / (5 * 5));

  // Seasonal windage
  const seasonalWind = seasonal.wind * windage;
  const uWind = seasonalWind * 2.5;
  const vWind = -seasonal.wind * 0.4 * windage;

  // Low-amplitude eddy noise (curl-free)
  const phase = (days / 28) * Math.PI * 2;
  const uEddy = 0.35 * Math.sin(phase + lon * 0.18) * Math.cos(lat * 0.2);
  const vEddy = 0.35 * Math.cos(phase + lon * 0.14) * Math.sin(lat * 0.16);

  const u = (uGyre + uJet + uAzores + uWind + uEddy) * 0.23 * speedFactor;
  const v = (vGyre + vJet + vCanary + vWind + vEddy) * 0.23 * speedFactor;

  const scalar =
    0.6 * Math.hypot(u, v) +
    0.4 * (Math.sin(lat * 0.4 + phase) * 0.5 + 0.5) * seasonal.gust;

  return { u, v, scalar };
}

export function integrateParticle(lon, lat, dtHours, tDays, options) {
  const h = dtHours / 24;
  const { u: u1, v: v1 } = velocityField(lon, lat, tDays, options);
  const dLon1 = ((u1 * dtHours * 3600) / metersPerDegLon(lat));
  const dLat1 = ((v1 * dtHours * 3600) / METERS_PER_DEG_LAT);

  const midLon = lon + dLon1 * 0.5;
  const midLat = lat + dLat1 * 0.5;
  const { u: u2, v: v2 } = velocityField(midLon, midLat, tDays + h * 0.5, options);
  const dLon2 = ((u2 * dtHours * 3600) / metersPerDegLon(midLat));
  const dLat2 = ((v2 * dtHours * 3600) / METERS_PER_DEG_LAT);

  const nextLon = lon + dLon2;
  const nextLat = lat + dLat2;

  return {
    lon: clamp(nextLon, DOMAIN.lonMin, DOMAIN.lonMax),
    lat: clamp(nextLat, DOMAIN.latMin, DOMAIN.latMax),
    u: u2,
    v: v2,
  };
}

export function createDensityGrid(nx = 192, ny = 128) {
  return {
    nx,
    ny,
    data: new Float32Array(nx * ny),
    accumHours: 0,
  };
}

export function resetDensityGrid(grid) {
  grid.data.fill(0);
  grid.accumHours = 0;
}

export function accumulateDensity(grid, lon, lat, mask) {
  const { nx, ny, data } = grid;
  const lonSpan = DOMAIN.lonMax - DOMAIN.lonMin;
  const latSpan = DOMAIN.latMax - DOMAIN.latMin;
  for (let i = 0; i < lon.length; i++) {
    if (!mask[i]) continue;
    const x = ((lon[i] - DOMAIN.lonMin) / lonSpan) * (nx - 1);
    const y = ((lat[i] - DOMAIN.latMin) / latSpan) * (ny - 1);
    const ix = x | 0;
    const iy = y | 0;
    if (ix < 0 || ix >= nx || iy < 0 || iy >= ny) continue;
    data[iy * nx + ix] += 1;
  }
}

export function computeBeachingProb(lon, lat, u, v, seasonFactor) {
  let prob = 0;
  for (const region of COAST_REGIONS) {
    if (
      lon >= region.lon0 &&
      lon <= region.lon1 &&
      lat >= region.lat0 &&
      lat <= region.lat1
    ) {
      const dot = u * region.normal[0] + v * region.normal[1];
      if (dot > 0) {
        prob += 0.012 * seasonFactor;
      }
    }
  }
  return prob > 0.25 ? 0.25 : prob;
}

export function stepParticles(pool, dtHours, tDays, options) {
  const {
    enableBeaching = true,
    season = "DJF",
    windage = 0.02,
    speedFactor = 1.0,
    accumulateHeatmap = false,
    densityGrid,
    rng = Math.random,
    maskBuffer,
  } = options;

  if (!pool.count) {
    return { beachedUS: 0, beachedEU: 0 };
  }

  const seasonal = SEASONS[season] || SEASONS.DJF;
  const mask = maskBuffer || new Array(pool.count);

  let beachedUS = 0;
  let beachedEU = 0;

  for (let i = 0; i < pool.count; i++) {
    if (pool.beached[i]) {
      mask[i] = false;
      continue;
    }

    const { lon, lat } = integrateParticle(
      pool.lon[i],
      pool.lat[i],
      dtHours,
      tDays,
      { season, windage, speedFactor }
    );

    pool.lon[i] = lon;
    pool.lat[i] = lat;

    const { u, v, scalar } = velocityField(lon, lat, tDays, {
      season,
      windage,
      speedFactor,
    });
    pool.u[i] = u;
    pool.v[i] = v;
    pool.scalar[i] = scalar;
    pool.ageHours[i] += dtHours;

    let beached = false;
    if (enableBeaching) {
      const pb = computeBeachingProb(lon, lat, u, v, seasonal.beach);
      if (pb > 0 && rng() < pb) {
        beached = true;
        pool.beached[i] = 1;
        if (pool.regionTag[i] === 0) beachedUS++;
        else beachedEU++;
      }
    }

    mask[i] = !beached;
  }

  if (accumulateHeatmap && densityGrid) {
    accumulateDensity(densityGrid, pool.lon.subarray(0, pool.count), pool.lat.subarray(0, pool.count), mask);
    densityGrid.accumHours += dtHours;
  }

  return { beachedUS, beachedEU };
}

export function estimateScalar(lon, lat, tDays, options) {
  return velocityField(lon, lat, tDays, options).scalar;
}

export function integrateTracker(startLon, startLat, params) {
  const {
    maxYears = 20,
    stepHours = 6,
    season = "DJF",
    windage = 0.02,
    speedFactor = 1.0,
  } = params;

  const positions = [];
  let lon = startLon;
  let lat = startLat;
  let tDays = 0;
  let distanceKm = 0;
  let openOceanHours = 0;
  const totalSteps = ((maxYears * 365 * 24) / stepHours) | 0;

  for (let i = 0; i < totalSteps; i++) {
    const current = integrateParticle(lon, lat, stepHours, tDays, {
      season,
      windage,
      speedFactor,
    });
    positions.push([current.lon, current.lat]);

    const segmentKm = haversineKm(lon, lat, current.lon, current.lat);
    distanceKm += segmentKm;
    const offshore = !isInShelf(current.lon, current.lat);
    if (offshore) {
      openOceanHours += stepHours;
    }

    lon = current.lon;
    lat = current.lat;
    tDays += stepHours / 24;
  }

  const openProbability = Math.min(1, openOceanHours / (maxYears * 365 * 24));

  return {
    positions,
    distanceKm,
    probability: openProbability,
  };
}

export function isInShelf(lon, lat) {
  return (
    (lon >= -81 && lon <= -66 && lat >= 24 && lat <= 44) ||
    (lon >= -12 && lon <= -3 && lat >= 35 && lat <= 46) ||
    (lon >= -4 && lon <= 2 && lat >= 45 && lat <= 51) ||
    (lon >= -8 && lon <= -2 && lat >= 50 && lat <= 56) ||
    (lon >= 5 && lon <= 12 && lat >= 55 && lat <= 60)
  );
}

export function haversineKm(lon0, lat0, lon1, lat1) {
  const R = 6371;
  const dLat = ((lat1 - lat0) * Math.PI) / 180;
  const dLon = ((lon1 - lon0) * Math.PI) / 180;
  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos((lat0 * Math.PI) / 180) *
      Math.cos((lat1 * Math.PI) / 180) *
      Math.sin(dLon / 2) *
      Math.sin(dLon / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return R * c;
}

export function sampleScalarGrid(nx, ny, tDays, options) {
  const grid = new Float32Array(nx * ny);
  const lonSpan = DOMAIN.lonMax - DOMAIN.lonMin;
  const latSpan = DOMAIN.latMax - DOMAIN.latMin;
  let ptr = 0;
  for (let j = 0; j < ny; j++) {
    const fracY = j / (ny - 1);
    const lat = DOMAIN.latMin + fracY * latSpan;
    for (let i = 0; i < nx; i++) {
      const fracX = i / (nx - 1);
      const lon = DOMAIN.lonMin + fracX * lonSpan;
      grid[ptr++] = estimateScalar(lon, lat, tDays, options);
    }
  }
  return grid;
}

export function normaliseScalarField(field) {
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < field.length; i++) {
    const v = field[i];
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const range = max - min || 1;
  const result = new Float32Array(field.length);
  for (let i = 0; i < field.length; i++) {
    result[i] = (field[i] - min) / range;
  }
  return { normalised: result, min, max };
}
