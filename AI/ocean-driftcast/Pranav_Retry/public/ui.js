// ui.js
// Control panel behaviour, state persistence, and DOM overlays

const STORAGE_KEY = "driftcast-ui-state-v1";

function defaultState() {
  return {
    mode: "currents",
    particles: 8000,
    speed: 1,
    windage: 0.02,
    season: "DJF",
    trails: true,
    heatmap: false,
    beaching: true,
    filters: {
      water: 1,
      population: 1,
      size: 1,
    },
  };
}

function loadState() {
  try {
    const stored = window.localStorage.getItem(STORAGE_KEY);
    if (!stored) return defaultState();
    const parsed = JSON.parse(stored);
    return { ...defaultState(), ...parsed };
  } catch (err) {
    console.warn("Failed to load DriftCast UI state", err);
    return defaultState();
  }
}

function saveState(state) {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  } catch (err) {
    console.warn("Failed to persist DriftCast UI state", err);
  }
}

function formatParticles(value) {
  if (value >= 1000) {
    return `${Math.round(value / 1000)}k`;
  }
  return String(value);
}

function formatSpeed(value) {
  return `${value.toFixed(2)}×`;
}

function formatWindage(value) {
  return value.toFixed(3);
}

export function initUI(options) {
  const state = loadState();
  const {
    onModeChange = () => {},
    onSliderChange = () => {},
    onToggleChange = () => {},
    onAction = () => {},
    onFilterChange = () => {},
    onPinSeed = () => {},
  } = options;

  const root = document.getElementById("control-panel");
  const app = document.getElementById("app");
  const hoverCard = document.getElementById("hover-card");

  const tabs = root.querySelectorAll(".tab-button");
  const sections = root.querySelectorAll(".mode-content");

  tabs.forEach((tab) => {
    const mode = tab.dataset.mode;
    tab.addEventListener("click", () => {
      if (state.mode === mode) return;
      setMode(mode);
      onModeChange(mode);
    });
  });

  function setMode(mode) {
    state.mode = mode;
    tabs.forEach((tab) => {
      const active = tab.dataset.mode === mode;
      tab.setAttribute("aria-selected", active ? "true" : "false");
    });
    sections.forEach((section) => {
      const active = section.dataset.mode === mode;
      section.hidden = !active;
    });
    saveState(state);
  }

  function bindRange(id, formatter, key) {
    const input = root.querySelector(id);
    const valueSpan = root.querySelector(`[data-bind='${key}']`);
    if (!input) return;
    input.value = state[key];
    valueSpan.textContent = formatter(Number(state[key]));

    input.addEventListener("input", () => {
      const value = Number(input.value);
      state[key] = value;
      valueSpan.textContent = formatter(value);
      saveState(state);
      onSliderChange(key, value);
    });
  }

  bindRange("#particles-range", formatParticles, "particles");
  bindRange("#speed-range", formatSpeed, "speed");
  bindRange("#windage-range", formatWindage, "windage");

  const seasonSelect = root.querySelector("#season-select");
  seasonSelect.value = state.season;
  seasonSelect.addEventListener("change", () => {
    state.season = seasonSelect.value;
    saveState(state);
    onSliderChange("season", state.season);
  });

  function bindToggle(id, key) {
    const el = root.querySelector(id);
    el.checked = Boolean(state[key]);
    el.addEventListener("change", () => {
      state[key] = el.checked;
      saveState(state);
      onToggleChange(key, el.checked);
    });
  }

  bindToggle("#toggle-trails", "trails");
  bindToggle("#toggle-heatmap", "heatmap");
  bindToggle("#toggle-beaching", "beaching");

  root.querySelectorAll("[data-action]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const action = btn.dataset.action;
      onAction(action);
    });
  });

  const filterInputs = root.querySelectorAll("[data-filter]");
  filterInputs.forEach((input) => {
    const key = input.dataset.filter;
    input.value = state.filters[key];
    input.addEventListener("input", () => {
      state.filters[key] = Number(input.value);
      saveState(state);
      onFilterChange({ ...state.filters });
    });
  });

  const metrics = {
    fps: root.querySelector("[data-bind='fps']"),
    active: root.querySelector("[data-bind='active']"),
    beachedUS: root.querySelector("[data-bind='beached-us']"),
    beachedEU: root.querySelector("[data-bind='beached-eu']"),
    trackerDistance: root.querySelector("[data-bind='tracker-distance']"),
    trackerProb: root.querySelector("[data-bind='tracker-prob']"),
    densityCaption: root.querySelector("[data-bind='density-caption']"),
  };

  const exportControls = {
    container: root.querySelector(".export-progress"),
    progress: root.querySelector(".export-progress progress"),
    stop: root.querySelector("[data-action='export-stop']"),
    linkWrap: root.querySelector(".export-link"),
    link: root.querySelector(".export-link a"),
  };

  const pinList = root.querySelector(".pin-list");
  const pinTemplate = document.getElementById("pin-card-template");
  const pinCards = new Map();
  const pinMarkers = new Map();

  function updateMetrics(values) {
    if (typeof values.fps === "number") metrics.fps.textContent = values.fps.toFixed(0);
    if (typeof values.active === "number") metrics.active.textContent = values.active.toLocaleString();
    if (typeof values.beachedUS === "number")
      metrics.beachedUS.textContent = values.beachedUS.toLocaleString();
    if (typeof values.beachedEU === "number")
      metrics.beachedEU.textContent = values.beachedEU.toLocaleString();
    if (typeof values.trackerDistance === "number")
      metrics.trackerDistance.textContent = `${values.trackerDistance.toFixed(0)} km`;
    if (typeof values.trackerProb === "number")
      metrics.trackerProb.textContent = `${Math.round(values.trackerProb * 100)}%`;
    if (typeof values.densityCaption === "string")
      metrics.densityCaption.textContent = values.densityCaption;
  }

  function renderPins(pins, projector) {
    pinList.innerHTML = "";
    pinCards.forEach((node) => node.remove());
    pinMarkers.forEach((node) => node.remove());
    pinCards.clear();
    pinMarkers.clear();

    for (const pin of pins) {
      const li = document.createElement("li");
      li.innerHTML = `<span>${pin.name}</span><button type="button">Seed</button>`;
      li.querySelector("button").addEventListener("click", () => onPinSeed(pin));
      pinList.appendChild(li);

      const marker = document.createElement("div");
      marker.className = `pin-marker pin-marker--${pin.region === "EU" ? "eu" : "us"}`;
      marker.title = pin.name;
      marker.dataset.pinName = pin.name;
      app.appendChild(marker);
      pinMarkers.set(pin.name, marker);

      marker.addEventListener("click", (ev) => {
        ev.stopPropagation();
        openPinCard(pin, marker);
      });

      positionMarker(pin, projector);
    }
  }

  function positionMarker(pin, projector) {
    const marker = pinMarkers.get(pin.name);
    if (!marker) return;
    const point = projector(pin.lon, pin.lat);
    if (!point) {
      marker.style.display = "none";
      return;
    }
    marker.style.display = "block";
    marker.style.left = `${point[0]}px`;
    marker.style.top = `${point[1]}px`;
  }

  function positionMarkers(pins, projector) {
    pins.forEach((pin) => positionMarker(pin, projector));
  }

  function openPinCard(pin, anchor) {
    closePinCards();
    const node = pinTemplate.content.cloneNode(true);
    const card = node.querySelector(".pin-card");
    card.querySelector("h2").textContent = pin.name;
    card.querySelector(".pin-card__meta").textContent = `${pin.region} release • weight ${pin.weight.toFixed(1)}`;
    card.querySelector(".pin-card__stats").innerHTML = `
      <div>Lon ${pin.lon.toFixed(2)}°</div>
      <div>Lat ${pin.lat.toFixed(2)}°</div>
      <div>Filters W ${pin.filters.water.toFixed(2)} • P ${pin.filters.population.toFixed(
      2
    )} • S ${pin.filters.size.toFixed(2)}</div>`;
    card.querySelector("[data-pin-action='seed']").addEventListener("click", () => {
      onPinSeed(pin);
    });
    document.body.appendChild(card);
    pinCards.set(pin.name, card);
    const rect = anchor.getBoundingClientRect();
    card.style.position = "absolute";
    card.style.left = `${rect.left + 24}px`;
    card.style.top = `${rect.top}px`;

    const closeOnOutside = (event) => {
      if (card.contains(event.target) || event.target === anchor) return;
      closePinCards();
      document.removeEventListener("mousedown", closeOnOutside);
    };
    document.addEventListener("mousedown", closeOnOutside);
  }

  function closePinCards() {
    pinCards.forEach((node) => node.remove());
    pinCards.clear();
  }

  function updateHoverCard(data) {
    if (!data) {
      hoverCard.style.display = "none";
      return;
    }
    const { x, y, lat, lon, dir, speed, scalar } = data;
    hoverCard.style.display = "grid";
    hoverCard.style.left = `${x + 18}px`;
    hoverCard.style.top = `${y + 18}px`;
    hoverCard.querySelector("[data-bind='lat']").textContent = lat.toFixed(2);
    hoverCard.querySelector("[data-bind='lon']").textContent = lon.toFixed(2);
    hoverCard.querySelector("[data-bind='dir']").textContent = `${Math.round(dir)}°`;
    hoverCard.querySelector("[data-bind='speed']").textContent = `${speed.toFixed(2)} m/s`;
    hoverCard.querySelector("[data-bind='scalar']").textContent = scalar.toFixed(2);
  }

  exportControls.container.hidden = true;
  exportControls.linkWrap.hidden = true;

  exportControls.stop.addEventListener("click", () => onAction("export-stop"));

  function showExportProgress(show) {
    exportControls.container.hidden = !show;
    if (show) {
      exportControls.linkWrap.hidden = true;
    }
  }

  function updateExportProgress(value) {
    exportControls.progress.value = value;
  }

  function showExportLink(href, filename) {
    exportControls.linkWrap.hidden = false;
    exportControls.link.href = href;
    exportControls.link.download = filename;
  }

  function updateDensityCaption(text) {
    metrics.densityCaption.textContent = text;
  }

  setMode(state.mode);
  onFilterChange({ ...state.filters });

  return {
    state,
    setMode,
    updateMetrics,
    updateHoverCard,
    renderPins,
    positionMarkers,
    closePinCards,
    showExportProgress,
    updateExportProgress,
    showExportLink,
    updateDensityCaption,
  };
}
