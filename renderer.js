const tf = require('@tensorflow/tfjs');
const osc = require('osc');
const { ipcRenderer } = require('electron');

// --- State ---
let model = null;
let classLabels = [];
let rois = []; // { id, name, x, y, w, h, color }
let roiIdCounter = 0;
let inferenceTimer = null;
let oscPort = null;
let oscConnected = false;
let latestResults = {}; // roiId -> { label, probability }

// Drag interaction state
let dragMode = 'none'; // 'none' | 'draw' | 'move' | 'resize'
let dragTarget = null;  // ROI being moved/resized
let dragStart = { x: 0, y: 0 }; // overlay pixel coords at mousedown
let dragAnchor = null;  // resize: normalized coords of the fixed corner
let dragOffset = { dx: 0, dy: 0 }; // move: offset from ROI origin to click point (normalized)
const CORNER_HIT_RADIUS = 10; // pixels

// ROI colors
const ROI_COLORS = [
  '#e94560', '#00d2ff', '#00e676', '#ffab00',
  '#d500f9', '#ff6e40', '#64ffda', '#eeff41',
  '#ea80fc', '#84ffff',
];

// --- DOM elements ---
const video = document.getElementById('webcam');
const overlay = document.getElementById('roi-overlay');
const ctx = overlay.getContext('2d');
const cameraSelect = document.getElementById('camera-select');
const modelUrlInput = document.getElementById('model-url');
const loadModelBtn = document.getElementById('load-model-btn');
const loadZipBtn = document.getElementById('load-zip-btn');
const modelStatus = document.getElementById('model-status');
const roiListEl = document.getElementById('roi-list');
const clearRoisBtn = document.getElementById('clear-rois-btn');
const oscHostInput = document.getElementById('osc-host');
const oscPortInput = document.getElementById('osc-port');
const oscPrefixInput = document.getElementById('osc-prefix');
const oscToggle = document.getElementById('osc-toggle');
const oscGearBtn = document.getElementById('osc-gear-btn');
const oscSettingsPopover = document.getElementById('osc-settings-popover');
const oscConnStatusEl = document.getElementById('osc-conn-status');
const inferenceIntervalInput = document.getElementById('inference-interval');
const oscMonitorEl = document.getElementById('osc-monitor');
const statusEl = document.getElementById('status') || { textContent: '' };

// --- Camera ---
async function enumerateCameras() {
  const devices = await navigator.mediaDevices.enumerateDevices();
  const videoDevices = devices.filter(d => d.kind === 'videoinput');
  cameraSelect.innerHTML = '';
  videoDevices.forEach((d, i) => {
    const opt = document.createElement('option');
    opt.value = d.deviceId;
    opt.textContent = d.label || `Camera ${i + 1}`;
    cameraSelect.appendChild(opt);
  });
  return videoDevices;
}

async function startCamera(deviceId) {
  const constraints = {
    video: deviceId ? { deviceId: { exact: deviceId } } : true,
  };
  try {
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    video.srcObject = stream;
    await video.play();
    resizeOverlay();
    statusEl.textContent = 'Camera started';
  } catch (err) {
    statusEl.textContent = `Camera error: ${err.message}`;
  }
}

function resizeOverlay() {
  const rect = video.getBoundingClientRect();
  overlay.width = rect.width;
  overlay.height = rect.height;
  overlay.style.width = rect.width + 'px';
  overlay.style.height = rect.height + 'px';
}

cameraSelect.addEventListener('change', () => startCamera(cameraSelect.value));
window.addEventListener('resize', () => {
  resizeOverlay();
  drawROIs();
});

// --- Video coordinate helpers ---
// Convert overlay pixel coords to normalized video coords (0-1)
function overlayToNorm(ox, oy) {
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  const ow = overlay.width;
  const oh = overlay.height;
  if (!vw || !vh) return { nx: 0, ny: 0 };

  const videoAspect = vw / vh;
  const overlayAspect = ow / oh;

  let renderW, renderH, offsetX, offsetY;
  if (overlayAspect > videoAspect) {
    renderH = oh;
    renderW = oh * videoAspect;
    offsetX = (ow - renderW) / 2;
    offsetY = 0;
  } else {
    renderW = ow;
    renderH = ow / videoAspect;
    offsetX = 0;
    offsetY = (oh - renderH) / 2;
  }

  const nx = (ox - offsetX) / renderW;
  const ny = (oy - offsetY) / renderH;
  return { nx: Math.max(0, Math.min(1, nx)), ny: Math.max(0, Math.min(1, ny)) };
}

// Convert normalized coords to overlay pixel coords
function normToOverlay(nx, ny) {
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  const ow = overlay.width;
  const oh = overlay.height;
  if (!vw || !vh) return { ox: 0, oy: 0 };

  const videoAspect = vw / vh;
  const overlayAspect = ow / oh;

  let renderW, renderH, offsetX, offsetY;
  if (overlayAspect > videoAspect) {
    renderH = oh;
    renderW = oh * videoAspect;
    offsetX = (ow - renderW) / 2;
    offsetY = 0;
  } else {
    renderW = ow;
    renderH = ow / videoAspect;
    offsetX = 0;
    offsetY = (oh - renderH) / 2;
  }

  return { ox: nx * renderW + offsetX, oy: ny * renderH + offsetY };
}

function normSizeToOverlay(nw, nh) {
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  const ow = overlay.width;
  const oh = overlay.height;
  if (!vw || !vh) return { sw: 0, sh: 0 };

  const videoAspect = vw / vh;
  const overlayAspect = ow / oh;

  let renderW, renderH;
  if (overlayAspect > videoAspect) {
    renderH = oh;
    renderW = oh * videoAspect;
  } else {
    renderW = ow;
    renderH = ow / videoAspect;
  }

  return { sw: nw * renderW, sh: nh * renderH };
}

// --- ROI drawing on overlay ---
function drawROIs(tempRect) {
  ctx.clearRect(0, 0, overlay.width, overlay.height);

  rois.forEach(roi => {
    const { ox, oy } = normToOverlay(roi.x, roi.y);
    const { sw, sh } = normSizeToOverlay(roi.w, roi.h);
    ctx.strokeStyle = roi.color;
    ctx.lineWidth = 2;
    ctx.strokeRect(ox, oy, sw, sh);
    ctx.fillStyle = roi.color + '30';
    ctx.fillRect(ox, oy, sw, sh);
    // Label with detection result
    const result = latestResults[roi.id];
    const displayName = roi.name || `roi${roi.id}`;
    const labelText = result
      ? `${displayName}: ${result.label} (${(result.probability * 100).toFixed(0)}%)`
      : displayName;
    // Background for readability
    ctx.font = 'bold 13px sans-serif';
    const textWidth = ctx.measureText(labelText).width;
    ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
    ctx.fillRect(ox, oy - 20, textWidth + 8, 20);
    ctx.fillStyle = roi.color;
    ctx.fillText(labelText, ox + 4, oy - 6);
    // Corner handles
    const handleSize = 6;
    ctx.fillStyle = roi.color;
    const corners = [
      [ox, oy], [ox + sw, oy],
      [ox, oy + sh], [ox + sw, oy + sh],
    ];
    corners.forEach(([cx, cy]) => {
      ctx.fillRect(cx - handleSize / 2, cy - handleSize / 2, handleSize, handleSize);
    });
  });

  if (tempRect) {
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 1;
    ctx.setLineDash([5, 5]);
    ctx.strokeRect(tempRect.x, tempRect.y, tempRect.w, tempRect.h);
    ctx.setLineDash([]);
  }
}

// --- Hit testing ---
// Returns { roi, cornerIndex } or null. cornerIndex: 0=TL, 1=TR, 2=BL, 3=BR
function hitTestCorner(px, py) {
  for (let i = rois.length - 1; i >= 0; i--) {
    const roi = rois[i];
    const { ox, oy } = normToOverlay(roi.x, roi.y);
    const { sw, sh } = normSizeToOverlay(roi.w, roi.h);
    const corners = [
      [ox, oy], [ox + sw, oy],
      [ox, oy + sh], [ox + sw, oy + sh],
    ];
    for (let ci = 0; ci < corners.length; ci++) {
      const [cx, cy] = corners[ci];
      if (Math.hypot(px - cx, py - cy) <= CORNER_HIT_RADIUS) {
        return { roi, cornerIndex: ci };
      }
    }
  }
  return null;
}

// Returns roi or null
function hitTestROI(px, py) {
  for (let i = rois.length - 1; i >= 0; i--) {
    const roi = rois[i];
    const { ox, oy } = normToOverlay(roi.x, roi.y);
    const { sw, sh } = normSizeToOverlay(roi.w, roi.h);
    if (px >= ox && px <= ox + sw && py >= oy && py <= oy + sh) {
      return roi;
    }
  }
  return null;
}

// Returns CSS cursor for given overlay position
function getCursorAt(px, py) {
  const corner = hitTestCorner(px, py);
  if (corner) {
    return (corner.cornerIndex === 0 || corner.cornerIndex === 3) ? 'nwse-resize' : 'nesw-resize';
  }
  if (hitTestROI(px, py)) return 'move';
  return 'crosshair';
}

// --- ROI mouse interaction ---
overlay.addEventListener('mousedown', (e) => {
  const rect = overlay.getBoundingClientRect();
  const px = e.clientX - rect.left;
  const py = e.clientY - rect.top;

  // 1. Check corner hit → resize
  const corner = hitTestCorner(px, py);
  if (corner) {
    dragMode = 'resize';
    dragTarget = corner.roi;
    // Anchor = the opposite corner (fixed point)
    const ci = corner.cornerIndex;
    dragAnchor = {
      nx: (ci === 1 || ci === 3) ? dragTarget.x : dragTarget.x + dragTarget.w,
      ny: (ci === 2 || ci === 3) ? dragTarget.y : dragTarget.y + dragTarget.h,
    };
    return;
  }

  // 2. Check ROI interior hit → move
  const hitRoi = hitTestROI(px, py);
  if (hitRoi) {
    dragMode = 'move';
    dragTarget = hitRoi;
    const clickNorm = overlayToNorm(px, py);
    dragOffset = { dx: clickNorm.nx - hitRoi.x, dy: clickNorm.ny - hitRoi.y };
    return;
  }

  // 3. Empty space → draw new ROI
  dragMode = 'draw';
  dragStart = { x: px, y: py };
});

overlay.addEventListener('mousemove', (e) => {
  const rect = overlay.getBoundingClientRect();
  const px = e.clientX - rect.left;
  const py = e.clientY - rect.top;

  // Update cursor when not dragging
  if (dragMode === 'none') {
    overlay.style.cursor = getCursorAt(px, py);
    return;
  }

  if (dragMode === 'draw') {
    const dx = px - dragStart.x;
    const dy = py - dragStart.y;
    const size = Math.max(Math.abs(dx), Math.abs(dy));
    const sx = dragStart.x + (dx < 0 ? -size : 0);
    const sy = dragStart.y + (dy < 0 ? -size : 0);
    const tempRect = { x: sx, y: sy, w: size, h: size };
    drawROIs(tempRect);
    return;
  }

  if (dragMode === 'move' && dragTarget) {
    const norm = overlayToNorm(px, py);
    let newX = norm.nx - dragOffset.dx;
    let newY = norm.ny - dragOffset.dy;
    // Clamp to 0-1
    newX = Math.max(0, Math.min(1 - dragTarget.w, newX));
    newY = Math.max(0, Math.min(1 - dragTarget.h, newY));
    dragTarget.x = newX;
    dragTarget.y = newY;
    drawROIs();
    updateROIList();
    return;
  }

  if (dragMode === 'resize' && dragTarget) {
    // Work in overlay pixel space for square constraint
    const { ox: anchorPx, oy: anchorPy } = normToOverlay(dragAnchor.nx, dragAnchor.ny);
    const dpx = px - anchorPx;
    const dpy = py - anchorPy;
    const side = Math.max(Math.abs(dpx), Math.abs(dpy));
    const sx = anchorPx + (dpx < 0 ? -side : 0);
    const sy = anchorPy + (dpy < 0 ? -side : 0);
    const tl = overlayToNorm(sx, sy);
    const br = overlayToNorm(sx + side, sy + side);
    dragTarget.x = tl.nx;
    dragTarget.y = tl.ny;
    dragTarget.w = br.nx - tl.nx;
    dragTarget.h = br.ny - tl.ny;
    drawROIs();
    updateROIList();
    return;
  }
});

overlay.addEventListener('mouseup', (e) => {
  if (dragMode === 'draw') {
    const rect = overlay.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;
    const dx = px - dragStart.x;
    const dy = py - dragStart.y;
    const size = Math.max(Math.abs(dx), Math.abs(dy));

    if (size >= 10) {
      const sx = dragStart.x + (dx < 0 ? -size : 0);
      const sy = dragStart.y + (dy < 0 ? -size : 0);
      const topLeft = overlayToNorm(sx, sy);
      const bottomRight = overlayToNorm(sx + size, sy + size);

      const newId = ++roiIdCounter;
      const roi = {
        id: newId,
        name: `roi${newId}`,
        x: topLeft.nx,
        y: topLeft.ny,
        w: bottomRight.nx - topLeft.nx,
        h: bottomRight.ny - topLeft.ny,
        color: ROI_COLORS[(newId - 1) % ROI_COLORS.length],
      };
      rois.push(roi);
      restartInference();
    }
  }

  // For move/resize, values are already updated in mousemove
  dragMode = 'none';
  dragTarget = null;
  dragAnchor = null;
  updateROIList();
  drawROIs();
});

overlay.addEventListener('mouseleave', () => {
  if (dragMode !== 'none') {
    dragMode = 'none';
    dragTarget = null;
    dragAnchor = null;
    drawROIs();
  }
});

// --- ROI list UI ---
function updateROIList() {
  roiListEl.innerHTML = '';
  rois.forEach(roi => {
    const div = document.createElement('div');
    div.className = 'roi-item';
    div.innerHTML = `
      <span class="roi-color" style="background:${roi.color}"></span>
      <input type="text" class="roi-name-input" value="${roi.name}" spellcheck="false">
      <button class="btn-danger" data-id="${roi.id}">X</button>
    `;
    div.querySelector('.roi-name-input').addEventListener('change', (e) => {
      roi.name = e.target.value.trim().replace(/\s+/g, '_') || `roi${roi.id}`;
      e.target.value = roi.name;
      clearOSCMonitor();
      drawROIs();
    });
    div.querySelector('.btn-danger').addEventListener('click', () => {
      delete latestResults[roi.id];
      rois = rois.filter(r => r.id !== roi.id);
      updateROIList();
      drawROIs();
      restartInference();
    });
    roiListEl.appendChild(div);
  });
}

clearRoisBtn.addEventListener('click', () => {
  rois = [];
  roiIdCounter = 0;
  updateROIList();
  drawROIs();
});

// --- Teachable Machine model loading ---
async function loadModelFromBase(base, useCache) {
  modelStatus.textContent = 'Loading model...';

  const fetchOpts = useCache ? {} : { cache: 'no-store' };
  const cacheBust = useCache ? '' : '?v=' + Date.now();

  // Load metadata
  const metaRes = await fetch(base + 'metadata.json' + cacheBust, fetchOpts);
  const metadata = await metaRes.json();
  classLabels = metadata.labels || [];
  modelStatus.textContent = `Classes: ${classLabels.join(', ')} — loading weights...`;

  // Load TF model
  const loadOpts = useCache ? {} : {
    fetchFunc: (input, init) => {
      const u = typeof input === 'string' ? input : input.url;
      const sep = u.includes('?') ? '&' : '?';
      return fetch(u + sep + 'v=' + Date.now(), { ...init, cache: 'no-store' });
    },
  };
  model = await tf.loadLayersModel(base + 'model.json' + cacheBust, loadOpts);
  modelStatus.textContent = `Loaded: ${classLabels.length} classes (${classLabels.join(', ')})`;
  statusEl.textContent = 'Model loaded';
  restartInference();
}

// Load from URL — toggle popover, then submit
const modelUrlPopover = document.getElementById('model-url-popover');
const modelUrlSubmit = document.getElementById('model-url-submit');

loadModelBtn.addEventListener('click', () => {
  modelUrlPopover.classList.toggle('open');
  if (modelUrlPopover.classList.contains('open')) {
    modelUrlInput.focus();
  }
});

modelUrlSubmit.addEventListener('click', async () => {
  const url = modelUrlInput.value.trim();
  if (!url) return;
  modelUrlSubmit.disabled = true;
  try {
    try { await ipcRenderer.invoke('clear-cache'); } catch (_) {}
    const base = url.endsWith('/') ? url : url + '/';
    await loadModelFromBase(base, false);
    modelUrlPopover.classList.remove('open');
  } catch (err) {
    modelStatus.textContent = `Error: ${err.message}`;
  }
  modelUrlSubmit.disabled = false;
});

// Load from ZIP
loadZipBtn.addEventListener('click', async () => {
  loadZipBtn.disabled = true;
  try {
    const extractDir = await ipcRenderer.invoke('select-model-zip');
    if (!extractDir) { loadZipBtn.disabled = false; return; }
    // Use file:// URL for local model
    const base = 'file://' + extractDir + '/';
    await loadModelFromBase(base, true);
  } catch (err) {
    modelStatus.textContent = `Error: ${err.message}`;
  }
  loadZipBtn.disabled = false;
});

// --- Inference ---
const cropCanvas = document.createElement('canvas');
cropCanvas.width = 224;
cropCanvas.height = 224;
const cropCtx = cropCanvas.getContext('2d');

async function runInference() {
  if (!model || rois.length === 0) return;

  const vw = video.videoWidth;
  const vh = video.videoHeight;
  if (!vw || !vh) return;

  const results = [];

  for (const roi of rois) {
    const sx = Math.round(roi.x * vw);
    const sy = Math.round(roi.y * vh);
    const sw = Math.round(roi.w * vw);
    const sh = Math.round(roi.h * vh);

    if (sw < 1 || sh < 1) continue;

    // Crop ROI to 224x224
    cropCtx.drawImage(video, sx, sy, sw, sh, 0, 0, 224, 224);

    // Run prediction
    const tensor = tf.tidy(() => {
      return tf.browser.fromPixels(cropCanvas)
        .toFloat()
        .div(127.5)
        .sub(1)
        .expandDims(0);
    });

    const prediction = await model.predict(tensor);
    const probabilities = await prediction.data();
    tensor.dispose();
    prediction.dispose();

    const roiResult = {
      roi,
      predictions: classLabels.map((label, i) => ({
        label,
        probability: probabilities[i] || 0,
      })),
    };
    results.push(roiResult);

    // Send OSC
    sendOSC(roi, roiResult.predictions);
  }

  // Store latest top predictions for overlay display
  results.forEach(({ roi, predictions }) => {
    const top = predictions.reduce((a, b) => a.probability > b.probability ? a : b);
    latestResults[roi.id] = { label: top.label, probability: top.probability };
  });

  drawROIs();
}

function restartInference() {
  if (inferenceTimer) clearInterval(inferenceTimer);
  const interval = parseInt(inferenceIntervalInput.value) || 200;
  if (model && rois.length > 0) {
    inferenceTimer = setInterval(runInference, interval);
    statusEl.textContent = `Running inference every ${interval}ms on ${rois.length} ROI(s)`;
  }
}

inferenceIntervalInput.addEventListener('change', restartInference);


// --- OSC ---
// Gear button toggles settings popover
oscGearBtn.addEventListener('click', () => {
  oscSettingsPopover.classList.toggle('open');
});

// Toggle connect/disconnect
function oscConnect() {
  const host = oscHostInput.value.trim();
  const port = parseInt(oscPortInput.value);
  try {
    oscPort = new osc.UDPPort({
      localAddress: '0.0.0.0',
      localPort: 0,
      remoteAddress: host,
      remotePort: port,
      metadata: true,
    });
    oscPort.open();
    oscPort.on('ready', () => {
      oscConnected = true;
      oscToggle.classList.add('active');
      oscConnStatusEl.textContent = `${host}:${port}`;
    });
    oscPort.on('error', (err) => {
      oscConnStatusEl.textContent = `Error: ${err.message}`;
    });
  } catch (err) {
    oscConnStatusEl.textContent = `Error: ${err.message}`;
  }
}

function oscDisconnect() {
  if (oscPort) { oscPort.close(); oscPort = null; }
  oscConnected = false;
  oscToggle.classList.remove('active');
  oscConnStatusEl.textContent = 'Disconnected';
}

oscToggle.addEventListener('click', () => {
  if (oscConnected) oscDisconnect();
  else oscConnect();
});

// OSC monitor state
const oscMonitorRows = {}; // address -> { row, checkbox, enabled }
const oscActivityEl = document.getElementById('osc-activity');
let activityResetTimer = null;

function flashActivity() {
  oscActivityEl.classList.add('on');
  if (activityResetTimer) clearTimeout(activityResetTimer);
  activityResetTimer = setTimeout(() => oscActivityEl.classList.remove('on'), 120);
}

function clearOSCMonitor() {
  oscMonitorEl.innerHTML = '';
  Object.keys(oscMonitorRows).forEach(k => delete oscMonitorRows[k]);
}

function updateOSCMonitor(address, displayValue) {
  let entry = oscMonitorRows[address];
  if (!entry) {
    const row = document.createElement('div');
    row.className = 'osc-row';
    row.innerHTML = `<input type="checkbox" checked><span class="osc-addr"></span><span class="osc-val"></span>`;
    row.querySelector('.osc-addr').textContent = address;
    const cb = row.querySelector('input[type="checkbox"]');
    entry = { row, checkbox: cb, enabled: true };
    cb.addEventListener('change', () => {
      entry.enabled = cb.checked;
      row.classList.toggle('disabled', !cb.checked);
    });
    oscMonitorEl.appendChild(row);
    oscMonitorRows[address] = entry;
  }
  entry.row.querySelector('.osc-val').textContent = displayValue;
}

function sendOSC(roi, predictions) {
  const prefix = oscPrefixInput.value.trim() || '/tm/roi';
  const roiName = roi.name || `roi${roi.id}`;
  const topPred = predictions.reduce((a, b) => a.probability > b.probability ? a : b);

  // Build all messages
  const messages = [];
  const classAddr = `${prefix}/${roiName}/class`;
  const confAddr = `${prefix}/${roiName}/confidence`;
  messages.push({ address: classAddr, args: [{ type: 's', value: topPred.label }], display: topPred.label });
  messages.push({ address: confAddr, args: [{ type: 'f', value: topPred.probability }], display: topPred.probability.toFixed(3) });
  predictions.forEach((p, i) => {
    messages.push({
      address: `${prefix}/${roiName}/prob/${i}`,
      args: [{ type: 's', value: p.label }, { type: 'f', value: p.probability }],
      display: `${p.label} ${p.probability.toFixed(3)}`,
    });
  });

  // Update monitor + send only checked addresses
  let sent = false;
  messages.forEach(msg => {
    updateOSCMonitor(msg.address, msg.display);
    if (oscConnected && oscPort) {
      const entry = oscMonitorRows[msg.address];
      if (entry && entry.enabled) {
        try { oscPort.send({ address: msg.address, args: msg.args }); sent = true; } catch (_) {}
      }
    }
  });
  if (sent) flashActivity();
}

// --- Init ---
async function init() {
  await enumerateCameras();
  if (cameraSelect.options.length > 0) {
    await startCamera(cameraSelect.options[0].value);
  }
}

init();
