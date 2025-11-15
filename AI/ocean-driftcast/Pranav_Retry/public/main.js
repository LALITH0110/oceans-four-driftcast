import {
  DOMAIN,
  RELEASE_US,
  RELEASE_EU,
  createParticlePool,
  clearParticlePool,
  spawnParticles,
  stepParticles,
  createDensityGrid,
  resetDensityGrid,
  createMulberry32,
  sampleScalarGrid,
  normaliseScalarField,
  integrateTracker,
  velocityField,
} from "./physics.js";

import {
  createProjection,
  drawBase,
  fadeCanvas,
  drawParticles,
  drawStreaks,
  drawBeached,
  drawTrackerPath,
  drawHeatmap,
  renderScalarOverlay,
  createScalarPalette,
  createHeatmapPalette,
} from "./draw.js";

import { initUI } from "./ui.js";

const NATURAL_EARTH_URL = "https://cdn.jsdelivr.net/npm/world-atlas@2/land-110m.json";
const SCALAR_GRID = { nx: 140, ny: 96 };
const SIM_DT_HOURS = 3;
const STEP_INTERVAL_MS = 16;
const EXPORT_DURATION_MS = 6 * 60 * 1000; // default 6 minutes

const d3Geo = window.d3;
const topojsonClient = window.topojson;

async function loadNaturalEarth() {
  try {
    const resp = await fetch(NATURAL_EARTH_URL, { mode: "cors" });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const json = await resp.json();
    const land = topojsonClient.feature(json, json.objects.land);
    const graticule = d3Geo.geoGraticule10();
    return { land, graticule };
  } catch (err) {
    console.warn("Falling back to synthetic coastline", err);
    return { land: null, graticule: d3Geo.geoGraticule10() };
  }
}

function rngFromString(str) {
  let hash = 2166136261;
  for (let i = 0; i < str.length; i++) {
    hash ^= str.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return createMulberry32(hash >>> 0);
}

function synthesisePins() {
  const result = [];
  for (const site of RELEASE_US) {
    const rng = rngFromString(site.name);
    result.push({
      ...site,
      region: "US",
      filters: {
        water: 0.4 + rng() * 0.6,
        population: 0.3 + rng() * 0.7,
        size: 0.4 + rng() * 0.6,
      },
    });
  }
  for (const site of RELEASE_EU) {
    const rng = rngFromString(site.name);
    result.push({
      ...site,
      region: "EU",
      filters: {
        water: 0.4 + rng() * 0.6,
        population: 0.3 + rng() * 0.7,
        size: 0.5 + rng() * 0.5,
      },
    });
  }
  return result;
}

function formatYears(hours) {
  const years = hours / (24 * 365);
  return years.toFixed(1);
}

function setupCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  canvas.width = width * dpr;
  canvas.height = height * dpr;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.scale(dpr, dpr);
  return { ctx, width, height, dpr };
}

function clearCanvas(ctx) {
  ctx.save();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.restore();
}

function projectorFactory(projection) {
  return (lon, lat) => {
    const point = projection([lon, lat]);
    return point ? [point[0], point[1]] : null;
  };
}

function pointerToCanvas(canvas, evt) {
  const rect = canvas.getBoundingClientRect();
  const x = evt.clientX - rect.left;
  const y = evt.clientY - rect.top;
  return { x, y };
}

function computeBearing(u, v) {
  const angle = (Math.atan2(u, v) * 180) / Math.PI;
  return (angle + 360) % 360;
}

async function main() {
  const baseCanvas = document.getElementById("base-canvas");
  const animCanvas = document.getElementById("anim-canvas");
  let baseCanvasState = setupCanvas(baseCanvas);
  let animCanvasState = setupCanvas(animCanvas);
  let baseCtx = baseCanvasState.ctx;
  let animCtx = animCanvasState.ctx;
  const overlayCanvas = document.createElement("canvas");
  overlayCanvas.width = animCanvas.width;
  overlayCanvas.height = animCanvas.height;
  const overlayCtx = overlayCanvas.getContext("2d");

  let projection = createProjection(baseCanvasState.width, baseCanvasState.height);
  let projector = projectorFactory(projection);

  const topo = await loadNaturalEarth();
  drawBase(baseCtx, topo, projection);

  const pool = createParticlePool(60000);
  const maskBuffer = new Uint8Array(pool.max);
  const densityGrid = createDensityGrid(196, 132);
  const scalarPalette = createScalarPalette();
  const heatmapPalette = createHeatmapPalette();
  const releasePins = synthesisePins();
  let filteredPins = releasePins.slice();
  const rng = createMulberry32(133742);

  const ui = initUI({
    onModeChange: (mode) => {
      state.mode = mode;
    },
    onSliderChange: (key, value) => {
      if (key === "particles") {
        state.targetParticles = value;
      } else if (key === "speed") {
        state.speedFactor = value;
      } else if (key === "windage") {
        state.windage = value;
      } else if (key === "season") {
        state.season = value;
      }
    },
    onToggleChange: (key, value) => {
      if (key === "trails") state.trails = value;
      if (key === "heatmap") state.heatmapToggle = value;
      if (key === "beaching") state.beaching = value;
    },
    onAction: (action) => handleAction(action),
    onFilterChange: (filters) => {
      state.filters = filters;
      filteredPins = applyFilters();
      ui.renderPins(filteredPins, projector);
    },
    onPinSeed: (pin) => {
      queueSeed(pin.region === "EU" ? "EU" : "US", Math.max(1000, Math.round(state.targetParticles / 8)));
    },
  });

  const state = {
    running: false,
    lastFrame: performance.now(),
    accumulator: 0,
    timeDays: 0,
    speedFactor: ui.state.speed,
    windage: ui.state.windage,
    season: ui.state.season,
    trails: ui.state.trails,
    heatmapToggle: ui.state.heatmap,
    beaching: ui.state.beaching,
    pool,
    maskBuffer,
    densityGrid,
    overlay: null,
    overlayMeta: { min: 0, max: 1 },
    overlayAge: -Infinity,
    overlayInterval: 2500,
    overlayCtx,
    frameTimes: [],
    fps: 0,
    beachedUS: 0,
    beachedEU: 0,
    targetParticles: ui.state.particles,
    pendingSeeds: [],
    filters: ui.state.filters,
    pins: releasePins,
    tracker: null,
    trackerPositions: [],
    trackerDistance: 0,
    trackerProbability: 0,
    recording: null,
  };

  filteredPins = applyFilters();
  ui.renderPins(filteredPins, projector);

  function applyFilters() {
    const { filters } = state;
    return state.pins.filter((pin) => {
      return (
        pin.filters.water <= filters.water + 1e-6 &&
        pin.filters.population <= filters.population + 1e-6 &&
        pin.filters.size <= filters.size + 1e-6
      );
    });
  }

  function queueSeed(which, amount) {
    state.pendingSeeds.push({ which, amount });
    state.running = true;
  }

  function handleAction(action) {
    switch (action) {
      case "start":
        state.running = true;
        break;
      case "pause":
        state.running = false;
        break;
      case "reset":
        resetSimulation();
        break;
      case "seed-us":
        queueSeed("US", state.targetParticles);
        break;
      case "seed-eu":
        queueSeed("EU", state.targetParticles);
        break;
      case "export":
        startRecording();
        break;
      case "export-stop":
        stopRecording();
        break;
      default:
        break;
    }
  }

  function resetSimulation() {
    clearParticlePool(pool);
    state.running = false;
    state.timeDays = 0;
    state.beachedUS = 0;
    state.beachedEU = 0;
    state.tracker = null;
    state.trackerPositions = [];
    state.trackerDistance = 0;
    state.trackerProbability = 0;
    resetDensityGrid(densityGrid);
    clearCanvas(animCtx);
    ui.updateMetrics({
      active: 0,
      beachedUS: 0,
      beachedEU: 0,
      trackerDistance: 0,
      trackerProb: 0,
      densityCaption: "Synthetic plastics after 0 years.",
    });
  }

  function startRecording() {
    if (state.recording || typeof MediaRecorder === "undefined") {
      return;
    }
    if (typeof OffscreenCanvas === "undefined" && !animCanvas.captureStream) {
      console.warn("Capture stream not supported");
      return;
    }
    const recordCanvas = document.createElement("canvas");
    recordCanvas.width = 1920;
    recordCanvas.height = 1080;
    const recordCtx = recordCanvas.getContext("2d");
    const stream = recordCanvas.captureStream(30);
    const recorder = new MediaRecorder(stream, { mimeType: "video/webm;codecs=vp9" });
    const chunks = [];
    recorder.ondataavailable = (evt) => {
      if (evt.data && evt.data.size > 0) {
        chunks.push(evt.data);
      }
    };
    recorder.onstop = () => {
      const blob = new Blob(chunks, { type: "video/webm" });
      const url = URL.createObjectURL(blob);
      const filename = `driftcast_${new Date().toISOString().slice(0, 16).replace(/[:T]/g, "-")}.webm`;
      ui.showExportProgress(false);
      ui.updateExportProgress(0);
      ui.showExportLink(url, filename);
    };
    recorder.start(1000);
    state.recording = {
      canvas: recordCanvas,
      ctx: recordCtx,
      recorder,
      chunks,
      started: performance.now(),
      duration: EXPORT_DURATION_MS,
    };
    ui.showExportProgress(true);
  }

  function stopRecording() {
    if (!state.recording) return;
    state.recording.recorder.stop();
    state.recording = null;
  }

  function updateProjection() {
    baseCanvasState = setupCanvas(baseCanvas);
    animCanvasState = setupCanvas(animCanvas);
    baseCtx = baseCanvasState.ctx;
    animCtx = animCanvasState.ctx;
    overlayCanvas.width = animCanvas.width;
    overlayCanvas.height = animCanvas.height;
    projection = createProjection(baseCanvasState.width, baseCanvasState.height);
    projector = projectorFactory(projection);
    drawBase(baseCtx, topo, projection);
    ui.positionMarkers(filteredPins, projector);
  }

  window.addEventListener("resize", () => {
    updateProjection();
  });

  animCanvas.addEventListener("pointermove", (evt) => {
    const { x, y } = pointerToCanvas(animCanvas, evt);
    const geo = projection.invert([x, y]);
    if (!geo) {
      ui.updateHoverCard(null);
      return;
    }
    const [lon, lat] = geo;
    const { u, v, scalar } = velocityField(lon, lat, state.timeDays, {
      season: state.season,
      windage: state.windage,
      speedFactor: state.speedFactor,
    });
    const dir = computeBearing(u, v);
    const speed = Math.hypot(u, v);
    ui.updateHoverCard({ x, y, lat, lon, dir, speed, scalar });
  });

  animCanvas.addEventListener("pointerleave", () => ui.updateHoverCard(null));

  animCanvas.addEventListener("click", (evt) => {
    if (state.mode !== "tracker") return;
    const { x, y } = pointerToCanvas(animCanvas, evt);
    const geo = projection.invert([x, y]);
    if (!geo) return;
    state.tracker = { start: geo };
    const { positions, distanceKm, probability } = integrateTracker(geo[0], geo[1], {
      maxYears: 20,
      season: state.season,
      windage: state.windage,
      speedFactor: state.speedFactor,
    });
    state.trackerPositions = positions;
    state.trackerDistance = distanceKm;
    state.trackerProbability = probability;
    ui.updateMetrics({
      trackerDistance: distanceKm,
      trackerProb: probability,
    });
  });

  function maybeUpdateOverlay(now) {
    if (now - state.overlayAge < state.overlayInterval) return;
    const field = sampleScalarGrid(SCALAR_GRID.nx, SCALAR_GRID.ny, state.timeDays, {
      season: state.season,
      windage: state.windage,
      speedFactor: state.speedFactor,
    });
    const { normalised, min, max } = normaliseScalarField(field);
    state.overlay = normalised;
    state.overlayMeta = { min, max };
    state.overlayAge = now;
  }

  function seedPending() {
    while (state.pendingSeeds.length) {
      const { which, amount } = state.pendingSeeds.shift();
      spawnParticles(pool, amount, which, rng);
    }
  }

  function updateMetrics() {
    ui.updateMetrics({
      fps: state.fps,
      active: pool.count - countBeached(pool),
      beachedUS: state.beachedUS,
      beachedEU: state.beachedEU,
      densityCaption: `Synthetic plastics after ${formatYears(densityGrid.accumHours)} years.`,
    });
  }

  function countBeached(p) {
    let beached = 0;
    for (let i = 0; i < p.count; i++) {
      if (p.beached[i]) beached++;
    }
    return beached;
  }

  function drawFrame() {
    const now = performance.now();
    const dt = Math.min(64, now - state.lastFrame);
    state.lastFrame = now;
    state.frameTimes.push(dt);
    if (state.frameTimes.length > 20) state.frameTimes.shift();
    const avg = state.frameTimes.reduce((a, b) => a + b, 0) / state.frameTimes.length;
    state.fps = avg ? 1000 / avg : 0;

    if (state.running) {
      state.accumulator += dt;
      seedPending();
      while (state.accumulator >= STEP_INTERVAL_MS) {
        state.accumulator -= STEP_INTERVAL_MS;
        state.timeDays += SIM_DT_HOURS / 24;
        const { beachedUS, beachedEU } = stepParticles(pool, SIM_DT_HOURS, state.timeDays, {
          enableBeaching: state.beaching,
          season: state.season,
          windage: state.windage,
          speedFactor: state.speedFactor,
          accumulateHeatmap: state.heatmapToggle || state.mode === "density",
          densityGrid,
          rng,
          maskBuffer,
        });
        state.beachedUS += beachedUS;
        state.beachedEU += beachedEU;
      }
    }

    if (state.trails) {
      fadeCanvas(animCtx, 0.12);
    } else {
      clearCanvas(animCtx);
    }

    maybeUpdateOverlay(now);

    if (state.mode === "currents") {
      if (state.overlay) {
        renderScalarOverlay(animCtx, state.overlay, SCALAR_GRID.nx, SCALAR_GRID.ny, scalarPalette, 0.45);
      }
      drawStreaks(animCtx, pool, projection);
    }

    if ((state.heatmapToggle || state.mode === "density") && densityGrid.accumHours > 0) {
      drawHeatmap(animCtx, densityGrid, heatmapPalette);
    }

    drawParticles(animCtx, pool, projection, { trails: state.trails });

    if (state.beaching) {
      drawBeached(animCtx, pool, projection);
    }

    if (state.trackerPositions.length) {
      drawTrackerPath(animCtx, state.trackerPositions, projection);
    }

    if (state.recording) {
      const { ctx, canvas, started, duration, recorder } = state.recording;
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(baseCanvas, 0, 0, canvas.width, canvas.height);
      ctx.drawImage(animCanvas, 0, 0, canvas.width, canvas.height);
      const progress = Math.min(1, (now - started) / duration);
      ui.updateExportProgress(progress);
      if (progress >= 1) {
        recorder.stop();
        state.recording = null;
      }
    }

    updateMetrics();
    requestAnimationFrame(drawFrame);
  }

  drawFrame();
}

main();
