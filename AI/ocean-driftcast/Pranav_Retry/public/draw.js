// draw.js
// Rendering helpers for DriftCast visual layers

import { DOMAIN } from "./physics.js";

const d3Geo = window.d3;

export function createProjection(width, height) {
  const projection = d3Geo
    .geoMercator()
    .center([(-100 + 20) / 2, (0 + 65) / 2])
    .scale(1)
    .translate([0, 0]);

  const path = d3Geo.geoPath(projection);
  const bounds = [[Infinity, Infinity], [-Infinity, -Infinity]];
  const corners = [
    [DOMAIN.lonMin, DOMAIN.latMin],
    [DOMAIN.lonMin, DOMAIN.latMax],
    [DOMAIN.lonMax, DOMAIN.latMin],
    [DOMAIN.lonMax, DOMAIN.latMax],
  ];

  for (const c of corners) {
    const p = projection(c);
    if (!p) continue;
    bounds[0][0] = Math.min(bounds[0][0], p[0]);
    bounds[0][1] = Math.min(bounds[0][1], p[1]);
    bounds[1][0] = Math.max(bounds[1][0], p[0]);
    bounds[1][1] = Math.max(bounds[1][1], p[1]);
  }

  const dx = bounds[1][0] - bounds[0][0];
  const dy = bounds[1][1] - bounds[0][1];
  const scale = 0.95 / Math.max(dx / width, dy / height);
  const translate = [
    width / 2 - scale * (bounds[0][0] + bounds[1][0]) / 2,
    height / 2 - scale * (bounds[0][1] + bounds[1][1]) / 2,
  ];

  return projection.scale(scale).translate(translate);
}

export function projectPoint(projection, lon, lat) {
  const p = projection([lon, lat]);
  if (!p) return null;
  return { x: p[0], y: p[1] };
}

export function drawBase(ctx, { land, graticule }, projection) {
  ctx.save();
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.lineWidth = 0.6;
  ctx.strokeStyle = "rgba(90,140,170,0.35)";

  if (graticule) {
    const path = d3Geo.geoPath(projection, ctx);
    ctx.beginPath();
    path(graticule);
    ctx.strokeStyle = "rgba(80,120,160,0.25)";
    ctx.stroke();
  }

  if (land) {
    const path = d3Geo.geoPath(projection, ctx);
    ctx.beginPath();
    path(land);
    ctx.fillStyle = "#0b1320";
    ctx.fill();
    ctx.strokeStyle = "rgba(120,160,190,0.32)";
    ctx.lineWidth = 0.9;
    ctx.stroke();
  } else {
    // rectangular fallback coastline
    ctx.strokeStyle = "rgba(120,160,190,0.28)";
    ctx.lineWidth = 1;
    const pad = 30;
    ctx.strokeRect(pad, pad, ctx.canvas.width - 2 * pad, ctx.canvas.height - 2 * pad);
  }

  ctx.restore();
}

export function renderScalarOverlay(ctx, field, nx, ny, palette, opacity = 0.6) {
  const canvas = ctx.canvas;
  const image = ctx.createImageData(canvas.width, canvas.height);
  const data = image.data;
  const width = canvas.width;
  const height = canvas.height;
  const scalarWidth = nx;
  const scalarHeight = ny;

  for (let y = 0; y < height; y++) {
    const v = (y / height) * (scalarHeight - 1);
    const v0 = Math.floor(v);
    const v1 = Math.min(scalarHeight - 1, v0 + 1);
    const fy = v - v0;

    for (let x = 0; x < width; x++) {
      const u = (x / width) * (scalarWidth - 1);
      const u0 = Math.floor(u);
      const u1 = Math.min(scalarWidth - 1, u0 + 1);
      const fx = u - u0;

      const idx00 = v0 * scalarWidth + u0;
      const idx01 = v0 * scalarWidth + u1;
      const idx10 = v1 * scalarWidth + u0;
      const idx11 = v1 * scalarWidth + u1;

      const f0 = field[idx00] * (1 - fx) + field[idx01] * fx;
      const f1 = field[idx10] * (1 - fx) + field[idx11] * fx;
      const value = f0 * (1 - fy) + f1 * fy;

      const color = palette(value).rgba();
      const ptr = (y * width + x) * 4;
      data[ptr] = color[0];
      data[ptr + 1] = color[1];
      data[ptr + 2] = color[2];
      data[ptr + 3] = Math.round(opacity * 255);
    }
  }

  ctx.putImageData(image, 0, 0);
}

export function fadeCanvas(ctx, amount = 0.1) {
  ctx.save();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.globalCompositeOperation = "destination-out";
  ctx.fillStyle = `rgba(0,0,0,${amount})`;
  ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.restore();
}

export function drawParticles(ctx, pool, projection, { trails = true }) {
  ctx.save();
  ctx.globalCompositeOperation = trails ? "lighter" : "source-over";
  ctx.fillStyle = "rgba(240,250,255,0.8)";
  const radius = trails ? 1.1 : 1.6;

  for (let i = 0; i < pool.count; i++) {
    if (pool.beached[i]) continue;
    const p = projection([pool.lon[i], pool.lat[i]]);
    if (!p) continue;
    ctx.beginPath();
    ctx.arc(p[0], p[1], radius, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.restore();
}

export function drawStreaks(ctx, pool, projection) {
  ctx.save();
  ctx.globalCompositeOperation = "lighter";
  ctx.lineWidth = 0.6;

  for (let i = 0; i < pool.count; i++) {
    if (pool.beached[i]) continue;
    const lon = pool.lon[i];
    const lat = pool.lat[i];
    const u = pool.u[i];
    const v = pool.v[i];

    const p0 = projection([lon, lat]);
    const p1 = projection([lon + u * 0.3, lat + v * 0.3]);
    if (!p0 || !p1) continue;
    const speed = Math.hypot(u, v);
    const light = Math.min(1, speed * 0.7 + 0.2);
    ctx.strokeStyle = `rgba(${140 + light * 90},${200 + light * 30},255,${0.25 + light * 0.3})`;

    ctx.beginPath();
    ctx.moveTo(p0[0], p0[1]);
    ctx.lineTo(p1[0], p1[1]);
    ctx.stroke();
  }

  ctx.restore();
}

export function drawBeached(ctx, pool, projection) {
  ctx.save();
  ctx.strokeStyle = "rgba(255,170,64,0.8)";
  ctx.lineWidth = 2;

  for (let i = 0; i < pool.count; i++) {
    if (!pool.beached[i]) continue;
    const p = projection([pool.lon[i], pool.lat[i]]);
    if (!p) continue;
    ctx.beginPath();
    ctx.moveTo(p[0] - 4, p[1]);
    ctx.lineTo(p[0] + 4, p[1]);
    ctx.stroke();
  }

  ctx.restore();
}

export function drawTrackerPath(ctx, positions, projection) {
  if (!positions || positions.length === 0) return;
  ctx.save();
  ctx.lineWidth = 4;
  ctx.strokeStyle = "rgba(73,255,167,0.9)";
  ctx.shadowColor = "rgba(73,255,167,0.6)";
  ctx.shadowBlur = 12;

  ctx.beginPath();
  const first = projection(positions[0]);
  if (!first) return;
  ctx.moveTo(first[0], first[1]);
  for (let i = 1; i < positions.length; i++) {
    const p = projection(positions[i]);
    if (!p) continue;
    ctx.lineTo(p[0], p[1]);
  }
  ctx.stroke();
  ctx.restore();
}

export function drawHeatmap(ctx, grid, palette) {
  if (!grid) return;
  const { nx, ny, data } = grid;
  const canvas = ctx.canvas;
  const image = ctx.createImageData(nx, ny);
  const arr = image.data;
  let max = 0;
  for (let i = 0; i < data.length; i++) {
    if (data[i] > max) max = data[i];
  }
  const scale = max > 0 ? 1 / max : 0;
  for (let i = 0; i < data.length; i++) {
    const v = data[i] * scale;
    const color = palette(clamp01(v)).rgba();
    const ptr = i * 4;
    arr[ptr] = color[0];
    arr[ptr + 1] = color[1];
    arr[ptr + 2] = color[2];
    arr[ptr + 3] = Math.round(200 * v);
  }

  const offscreen =
    typeof OffscreenCanvas !== "undefined"
      ? new OffscreenCanvas(nx, ny)
      : Object.assign(document.createElement("canvas"), { width: nx, height: ny });
  const octx = offscreen.getContext("2d");
  octx.putImageData(image, 0, 0);

  ctx.save();
  ctx.globalAlpha = 0.8;
  ctx.imageSmoothingEnabled = true;
  ctx.filter = "blur(8px)";
  ctx.drawImage(offscreen, 0, 0, canvas.width, canvas.height);
  ctx.restore();
}

export function createScalarPalette() {
  return window.chroma
    .scale(["#112a56", "#1e90ff", "#8ef1ff", "#ffb64d", "#ff4d4f"])
    .mode("lab")
    .domain([0, 1]);
}

export function createHeatmapPalette() {
  return window.chroma
    .scale(["#04263a", "#146b6e", "#1fc678", "#f2e95d", "#ff5c2b"])
    .mode("lab")
    .domain([0, 1]);
}

export function clamp01(v) {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}
