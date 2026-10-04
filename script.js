/* ==========================================================================
   Air Canvas — script.js
   All state, gesture logic and drawing logic lives here.

   Layering (inside .stage):
     1. <video>        live mirrored camera feed
     2. #drawCanvas    permanent strokes — only cleared by Clear or a resize-replay
     3. #cursorCanvas  transient reticle — wiped and redrawn EVERY frame
   ========================================================================== */

import {
  HandLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

/* --------------------------------------------------------------------------
   DOM handles
   -------------------------------------------------------------------------- */
const video        = document.getElementById("video");
const stage        = document.getElementById("stage");
const drawCanvas   = document.getElementById("drawCanvas");
const cursorCanvas = document.getElementById("cursorCanvas");

const overlay        = document.getElementById("overlay");
const overlayTitle   = document.getElementById("overlayTitle");
const overlayText    = document.getElementById("overlayText");
const overlaySpinner = document.getElementById("overlaySpinner");

const pillModel  = document.getElementById("pillModel");
const pillHand   = document.getElementById("pillHand");
const pillTool   = document.getElementById("pillTool");
const modelValue = document.getElementById("modelValue");
const handValue  = document.getElementById("handValue");
const toolValue  = document.getElementById("toolValue");

const toolSelector  = document.getElementById("toolSelector");
const drawToggle    = document.getElementById("drawToggle");
const brushSlider   = document.getElementById("brushSlider");
const brushValue    = document.getElementById("brushValue");
const clearBtn      = document.getElementById("clearBtn");
const palette       = document.getElementById("palette");
const cameraSelect  = document.getElementById("cameraSelect");

const drawCtx   = drawCanvas.getContext("2d");
const cursorCtx = cursorCanvas.getContext("2d");

/* --------------------------------------------------------------------------
   Tunables
   -------------------------------------------------------------------------- */
const SMOOTH          = 0.45;  // EMA weight given to the newest sample
const DEBOUNCE_FRAMES = 4;     // consecutive frames needed to commit a gesture
const ANGLE_EXTENDED  = 150;   // PIP angle (deg) at/above which a finger is straight
const ANGLE_CURLED    = 100;   // PIP angle (deg) at/below which a finger is folded
const TAU             = Math.PI * 2;

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";

/* --------------------------------------------------------------------------
   State
   -------------------------------------------------------------------------- */
const state = {
  // Every stroke ever drawn, in chronological order. Each stroke stores
  // NORMALISED (0–1) points, never raw pixels — see replayStrokes().
  //   pen / eraser : { type, color, width, points:[{x,y}] }
  //   flower       : { type, color, baseRadius, points:[{x,y,rotation,scale}] }
  strokes: [],

  currentStroke: null,   // stroke currently being painted
  lastStamp: null,       // canvas-px {x,y,r} of the last flower stamp (spacing check)

  color: "#ffffff",
  tool: "pen",           // 'pen' | 'flower' — only affects the POINTING gesture
  brush: 8,              // shared brush-size slider value
  drawing: true,         // master ON/OFF

  mode: "idle",          // committed gesture: 'pen' | 'eraser' | 'idle'
  candidate: "idle",     // gesture waiting for debounce confirmation
  candidateCount: 0,

  smoothed: null,        // {x,y} normalised (raw, UNMIRRORED) EMA position
  projection: null,      // current object-fit:cover mapping
  handPresent: false,
};

let landmarker    = null;
let lastVideoTime = -1;
let rafId         = null;

/* ==========================================================================
   1. COORDINATE MAPPING  (the part that has to be exactly right)

   MediaPipe reports landmarks normalised 0–1 against the RAW, UNMIRRORED,
   UNCROPPED video frame. The video is displayed with `object-fit: cover`
   (which crops) and flipped with `transform: scaleX(-1)`. To land a stroke
   under the real fingertip we must:

     a) reproduce the cover math  → drawW/drawH + centering offsets
     b) map  x = offsetX + nx*drawW ,  y = offsetY + ny*drawH
     c) mirror:  finalX = canvasWidth - x

   Everything below is expressed in CSS pixels; the dpr scale is applied to
   the 2D context transform so we never have to think about it again.
   ========================================================================== */

function computeProjection() {
  // Intrinsic video size (fall back to a sane default before metadata lands)
  const vw = video.videoWidth  || 1280;
  const vh = video.videoHeight || 720;

  // Rendered size of the stage
  const cw = stage.clientWidth  || 1;
  const ch = stage.clientHeight || 1;

  // --- object-fit: cover, reproduced in JS ---
  // scale is the LARGER of the two ratios, so the frame always fills the box
  // and overflows on one axis (which gets cropped by the centering offsets).
  const scale = Math.max(cw / vw, ch / vh);
  const drawW = vw * scale;
  const drawH = vh * scale;

  return {
    cw, ch,
    drawW, drawH,
    offsetX: (cw - drawW) / 2,   // negative when the frame is wider than the box
    offsetY: (ch - drawH) / 2,
    aspect: vw / vh,             // used to correct landmark angles
  };
}

/** Normalised (raw, unmirrored) landmark → canvas CSS pixels (mirrored). */
function project(nx, ny) {
  const p = state.projection;
  const x = p.offsetX + nx * p.drawW;
  const y = p.offsetY + ny * p.drawH;
  return { x: p.cw - x, y };      // horizontal mirror to match scaleX(-1)
}

/* ==========================================================================
   2. CANVAS SIZING + RESIZE-REPLAY

   Why strokes are stored as NORMALISED points:
   the canvas backing store is destroyed every time the window resizes. If we
   stored raw pixels, every stroke would jump or stretch. Storing 0–1 points
   means we can re-project the whole history through the *current* cover math
   and everything lands exactly where it belongs — including eraser strokes,
   which must be replayed in order so they only erase what came before them.
   ========================================================================== */

let resizeQueued = false;

function queueResize() {
  if (resizeQueued) return;
  resizeQueued = true;
  requestAnimationFrame(() => {
    resizeQueued = false;
    resizeCanvases();
  });
}

function resizeCanvases() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = stage.clientWidth;
  const h = stage.clientHeight;

  for (const c of [drawCanvas, cursorCanvas]) {
    // Setting width/height wipes the backing store — that's expected here,
    // we replay immediately below.
    c.width  = Math.max(1, Math.round(w * dpr));
    c.height = Math.max(1, Math.round(h * dpr));
  }

  // Work in CSS pixels from now on.
  drawCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  cursorCtx.setTransform(dpr, 0, 0, dpr, 0, 0);

  state.projection = computeProjection();
  state.lastStamp = null;   // old canvas-px stamp position is meaningless now
  replayStrokes();
}

/** Repaint the ENTIRE stroke history in original chronological order. */
function replayStrokes() {
  const p = state.projection;
  if (!p) return;

  drawCtx.save();
  drawCtx.setTransform(1, 0, 0, 1, 0, 0);
  drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
  drawCtx.restore();

  for (const stroke of state.strokes) {
    paintWholeStroke(stroke);
  }
}

function paintWholeStroke(stroke) {
  if (stroke.type === "flower") {
    drawCtx.globalCompositeOperation = "source-over";
    for (const pt of stroke.points) {
      const c = project(pt.x, pt.y);
      // rotation + scale were stored on the point at stamp time, so the
      // replay reproduces the identical flower instead of re-randomising it.
      drawFlower(drawCtx, c.x, c.y, stroke.baseRadius, pt.rotation, pt.scale, stroke.color);
    }
    return;
  }

  // pen / eraser — same pipeline, different composite mode
  drawCtx.globalCompositeOperation =
    stroke.type === "eraser" ? "destination-out" : "source-over";
  drawCtx.lineWidth   = stroke.width;
  drawCtx.lineCap     = "round";
  drawCtx.lineJoin    = "round";
  drawCtx.strokeStyle = stroke.color;
  drawCtx.fillStyle   = stroke.color;

  const pts = stroke.points;
  if (!pts.length) return;

  if (pts.length === 1) {
    const c = project(pts[0].x, pts[0].y);
    drawCtx.beginPath();
    drawCtx.arc(c.x, c.y, stroke.width / 2, 0, TAU);
    drawCtx.fill();
  } else {
    drawCtx.beginPath();
    for (let i = 0; i < pts.length; i++) {
      const c = project(pts[i].x, pts[i].y);
      if (i === 0) drawCtx.moveTo(c.x, c.y);
      else         drawCtx.lineTo(c.x, c.y);
    }
    drawCtx.stroke();
  }

  drawCtx.globalCompositeOperation = "source-over";
}

/* ==========================================================================
   3. INCREMENTAL PAINTING (only the brand-new segment / stamp, for speed)
   ========================================================================== */

function paintLastSegment(stroke) {
  const pts = stroke.points;
  const n = pts.length;

  drawCtx.globalCompositeOperation =
    stroke.type === "eraser" ? "destination-out" : "source-over";
  drawCtx.lineWidth   = stroke.width;
  drawCtx.lineCap     = "round";
  drawCtx.lineJoin    = "round";
  drawCtx.strokeStyle = stroke.color;
  drawCtx.fillStyle   = stroke.color;

  if (n === 1) {
    const c = project(pts[0].x, pts[0].y);
    drawCtx.beginPath();
    drawCtx.arc(c.x, c.y, stroke.width / 2, 0, TAU);
    drawCtx.fill();
  } else {
    const a = project(pts[n - 2].x, pts[n - 2].y);
    const b = project(pts[n - 1].x, pts[n - 1].y);
    drawCtx.beginPath();
    drawCtx.moveTo(a.x, a.y);
    drawCtx.lineTo(b.x, b.y);
    drawCtx.stroke();
  }

  drawCtx.globalCompositeOperation = "source-over";
}

/* --------------------------------------------------------------------------
   Flower stamping
   -------------------------------------------------------------------------- */

function drawFlower(ctx, x, y, radius, rotation, scale, color) {
  const r = radius * scale;
  const petals   = 6;
  const petalLen = r * 0.52;   // ellipse vertical radius
  const petalW   = r * 0.30;

  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(rotation);
  ctx.fillStyle = color;

  for (let i = 0; i < petals; i++) {
    ctx.save();
    ctx.rotate((i / petals) * TAU);
    ctx.beginPath();
    // centre pushed out to -petalLen so the petal spans 0 → -2·petalLen
    ctx.ellipse(0, -petalLen, petalW, petalLen, 0, 0, TAU);
    ctx.fill();
    ctx.restore();
  }

  // stamen — contrast colour derived from luminance, not plain black/white
  ctx.beginPath();
  ctx.arc(0, 0, r * 0.24, 0, TAU);
  ctx.fillStyle = contrastColor(color);
  ctx.fill();

  ctx.restore();
}

function hexToRgb(hex) {
  let h = String(hex).replace("#", "");
  if (h.length === 3) h = h.split("").map((c) => c + c).join("");
  const n = parseInt(h, 16);
  return { r: (n >> 16) & 255, g: (n >> 8) & 255, b: n & 255 };
}

/** YIQ luminance → warm-dark centre on light petals, warm-gold on dark ones. */
function contrastColor(hex) {
  const { r, g, b } = hexToRgb(hex);
  const yiq = (r * 299 + g * 587 + b * 114) / 1000;
  return yiq >= 128 ? "#6b3a12" : "#ffd98a";
}

function hexA(hex, a) {
  const { r, g, b } = hexToRgb(hex);
  return `rgba(${r},${g},${b},${a})`;
}

/* --------------------------------------------------------------------------
   Size helpers — one slider drives all three tools
   -------------------------------------------------------------------------- */
const penWidth     = () => state.brush;
const eraserWidth  = () => state.brush * 4;     // eraser is much fatter
const flowerRadius = () => state.brush * 1.8;   // flower scaled up from brush

/* ==========================================================================
   4. GESTURE RECOGNITION

   We classify by JOINT ANGLE, not by raw y-coordinates. Comparing y values
   only works when the hand is upright; measuring the angle at the PIP joint
   between PIP→MCP and PIP→TIP works at any hand rotation.
   ========================================================================== */

const FINGERS = [
  { mcp: 5,  pip: 6,  tip: 8  }, // index
  { mcp: 9,  pip: 10, tip: 12 }, // middle
  { mcp: 13, pip: 14, tip: 16 }, // ring
  { mcp: 17, pip: 18, tip: 20 }, // pinky
];

/** Angle at vertex b, formed by a–b–c. */
function angleAt(a, b, c) {
  const v1x = a.x - b.x, v1y = a.y - b.y;
  const v2x = c.x - b.x, v2y = c.y - b.y;
  const m1 = Math.hypot(v1x, v1y);
  const m2 = Math.hypot(v2x, v2y);
  if (m1 < 1e-6 || m2 < 1e-6) return 180;
  let cos = (v1x * v2x + v1y * v2y) / (m1 * m2);
  cos = Math.max(-1, Math.min(1, cos));
  return (Math.acos(cos) * 180) / Math.PI;
}

/**
 * @param {Array} lms    21 MediaPipe landmarks (normalised)
 * @param {number} aspect videoWidth / videoHeight
 * @returns {'pen'|'eraser'|'idle'}
 */
function classifyGesture(lms, aspect) {
  // MediaPipe normalises x by WIDTH and y by HEIGHT. Multiplying x by the
  // aspect ratio restores isotropic space, so the angles we measure are the
  // real anatomical ones and don't change when the video aspect changes.
  const P = (i) => ({ x: lms[i].x * aspect, y: lms[i].y });

  const extended = FINGERS.map((f) => angleAt(P(f.mcp), P(f.pip), P(f.tip)) >= ANGLE_EXTENDED);
  const curled   = FINGERS.map((f) => angleAt(P(f.mcp), P(f.pip), P(f.tip)) <= ANGLE_CURLED);

  const [idx, mid, ring, pinky] = extended;
  const [cIdx, cMid, cRing, cPinky] = curled;

  // pointing: index out, all others folded  → pen
  if (idx && !mid && !ring && !pinky) return "pen";

  // fist: everything folded → eraser. We accept either "all clearly curled"
  // or the looser "nothing is extended" so a slightly sloppy fist still works.
  if (cIdx && cMid && cRing && cPinky) return "eraser";
  if (!idx && !mid && !ring && !pinky) return "eraser";

  // open palm: everything straight → idle (pause)
  if (idx && mid && ring && pinky) return "idle";

  // anything ambiguous → idle is the safest fallback (never draws by accident)
  return "idle";
}

/** Require the same raw gesture for N consecutive frames before committing. */
function updateDebounce(raw) {
  if (raw === state.mode) {
    state.candidate = raw;
    state.candidateCount = 0;
    return;
  }
  if (raw === state.candidate) {
    state.candidateCount++;
  } else {
    state.candidate = raw;
    state.candidateCount = 1;
  }
  if (state.candidateCount >= DEBOUNCE_FRAMES) {
    commitMode(raw);
    state.candidateCount = 0;
  }
}

function commitMode(mode) {
  endStroke();           // gestures must never bleed into each other
  state.mode = mode;
  state.lastStamp = null;
  updateToolPill();
}

/* ==========================================================================
   5. STROKE LIFECYCLE
   ========================================================================== */

/**
 * One shared "end current stroke" — called from all three triggers:
 *   • the hand disappears
 *   • a debounced gesture-mode change (pen ↔ eraser ↔ idle)
 *   • a manual Pen ↔ Flower switch
 * Nulling `smoothed` drops the "previous point", so a new stroke can never
 * connect back to where the last one finished.
 */
function endStroke() {
  if (state.currentStroke && state.currentStroke.points.length) {
    state.strokes.push(state.currentStroke);
  }
  state.currentStroke = null;
  state.smoothed = null;
  state.lastStamp = null;
}

function ensureStroke(kind) {
  if (state.currentStroke && state.currentStroke.type === kind) return;
  if (state.currentStroke) endStroke();

  if (kind === "flower") {
    state.currentStroke = {
      type: "flower",
      color: state.color,
      baseRadius: flowerRadius(),
      points: [],
    };
  } else {
    state.currentStroke = {
      type: kind,                                             // 'pen' | 'eraser'
      color: kind === "eraser" ? "#000000" : state.color,
      width: kind === "eraser" ? eraserWidth() : penWidth(),
      points: [],
    };
  }
  state.lastStamp = null;
}

function appendPoint(cur) {
  const s = state.currentStroke;
  if (!s) return;
  // store NORMALISED coords, not the projected pixels
  s.points.push({ x: state.smoothed.x, y: state.smoothed.y });
  paintLastSegment(s);
}

/**
 * Flower stamping + spacing check.
 * Without a minimum spacing we'd stamp on every animation frame (~60/s) and
 * the flowers would pile into a solid blob. We only stamp once the smoothed
 * cursor has travelled `radius × 1.3` canvas pixels since the last stamp.
 */
function maybeStampFlower(cur) {
  const s = state.currentStroke;
  if (!s) return;

  const last = state.lastStamp;
  if (last) {
    const d = Math.hypot(cur.x - last.x, cur.y - last.y);
    if (d < last.r * 1.3) return;    // not far enough yet — skip this frame
  }

  const scale    = 0.85 + Math.random() * 0.3;   // ±15 %
  const rotation = Math.random() * TAU;          // full range

  s.points.push({
    x: state.smoothed.x,
    y: state.smoothed.y,
    rotation,
    scale,
  });

  drawFlower(drawCtx, cur.x, cur.y, s.baseRadius, rotation, scale, s.color);
  state.lastStamp = { x: cur.x, y: cur.y, r: s.baseRadius * scale };
}

/* ==========================================================================
   6. PER-FRAME CURSOR RENDERING (cursor canvas is wiped every frame)
   ========================================================================== */

function drawCursor(cur) {
  const p = state.projection;
  if (!p) return;

  cursorCtx.clearRect(0, 0, p.cw, p.ch);
  cursorCtx.save();

  // --- idle / paused: dim dashed reticle -----------------------------------
  if (!state.drawing || state.mode === "idle") {
    cursorCtx.beginPath();
    cursorCtx.arc(cur.x, cur.y, 9, 0, TAU);
    cursorCtx.strokeStyle = "rgba(255,255,255,0.35)";
    cursorCtx.lineWidth = 1.5;
    cursorCtx.setLineDash([3, 4]);
    cursorCtx.stroke();
    cursorCtx.setLineDash([]);
    cursorCtx.restore();
    return;
  }

  // --- eraser: dashed ring sized to the real eraser footprint ---------------
  if (state.mode === "eraser") {
    cursorCtx.beginPath();
    cursorCtx.arc(cur.x, cur.y, eraserWidth() / 2, 0, TAU);
    cursorCtx.strokeStyle = "rgba(255,255,255,0.85)";
    cursorCtx.lineWidth = 2;
    cursorCtx.setLineDash([6, 6]);
    cursorCtx.lineDashOffset = -performance.now() / 40;   // marching ants
    cursorCtx.stroke();
    cursorCtx.setLineDash([]);
    cursorCtx.lineDashOffset = 0;
    cursorCtx.restore();
    return;
  }

  // --- flower: translucent ghost preview of the NEXT stamp -------------------
  if (state.tool === "flower") {
    cursorCtx.globalAlpha = 0.38;
    drawFlower(cursorCtx, cur.x, cur.y, flowerRadius(), performance.now() / 1600, 1, state.color);
    cursorCtx.globalAlpha = 1;
    cursorCtx.restore();
    return;
  }

  // --- pen: glowing dot sized to the brush -----------------------------------
  const r = Math.max(3, penWidth() / 2);

  const g = cursorCtx.createRadialGradient(cur.x, cur.y, 0, cur.x, cur.y, r * 3.2);
  g.addColorStop(0, hexA(state.color, 0.55));
  g.addColorStop(1, hexA(state.color, 0));
  cursorCtx.fillStyle = g;
  cursorCtx.beginPath();
  cursorCtx.arc(cur.x, cur.y, r * 3.2, 0, TAU);
  cursorCtx.fill();

  cursorCtx.beginPath();
  cursorCtx.arc(cur.x, cur.y, r, 0, TAU);
  cursorCtx.fillStyle = state.color;
  cursorCtx.fill();

  cursorCtx.beginPath();
  cursorCtx.arc(cur.x, cur.y, r + 3, 0, TAU);
  cursorCtx.strokeStyle = "rgba(255,255,255,0.7)";
  cursorCtx.lineWidth = 1.5;
  cursorCtx.stroke();

  cursorCtx.restore();
}

/* ==========================================================================
   7. DETECTION LOOP
   ========================================================================== */

function loop() {
  rafId = requestAnimationFrame(loop);

  if (!landmarker || video.readyState < 2) return;
  if (video.currentTime === lastVideoTime) return;  // no new frame decoded yet
  lastVideoTime = video.currentTime;

  let result;
  try {
    result = landmarker.detectForVideo(video, performance.now());
  } catch (err) {
    console.warn("detectForVideo failed:", err);
    return;
  }
  handleResult(result);
}

function handleResult(result) {
  const p = state.projection;
  if (!p) return;

  const lms = result && result.landmarks && result.landmarks[0];

  /* ---- no hand ---------------------------------------------------------- */
  if (!lms) {
    if (state.handPresent) {
      // stop immediately, forget the debounce state, hide the reticle
      endStroke();
      state.mode = "idle";
      state.candidate = "idle";
      state.candidateCount = 0;
    }
    state.handPresent = false;
    updateHandPill(false);
    updateToolPill();
    cursorCtx.clearRect(0, 0, p.cw, p.ch);
    return;
  }

  /* ---- hand present ----------------------------------------------------- */
  state.handPresent = true;
  updateHandPill(true);

  const raw = classifyGesture(lms, p.aspect);
  updateDebounce(raw);

  /* ---- tracking point ---------------------------------------------------- */
  // The eraser follows the PALM CENTRE (mean of the wrist + the four MCP
  // knuckles) rather than the index tip: in a fist the index tip folds back
  // against the palm and its landmark jitters wildly, while the palm centre
  // stays stable.
  let nx, ny;
  if (state.mode === "eraser") {
    const ids = [0, 5, 9, 13, 17];
    let sx = 0, sy = 0;
    for (const i of ids) { sx += lms[i].x; sy += lms[i].y; }
    nx = sx / ids.length;
    ny = sy / ids.length;
  } else {
    nx = lms[8].x;   // index fingertip
    ny = lms[8].y;
  }

  // exponential moving average — takes the shake out of the cursor
  if (!state.smoothed) {
    state.smoothed = { x: nx, y: ny };
  } else {
    state.smoothed.x += (nx - state.smoothed.x) * SMOOTH;
    state.smoothed.y += (ny - state.smoothed.y) * SMOOTH;
  }

  const cur = project(state.smoothed.x, state.smoothed.y);

  /* ---- painting ---------------------------------------------------------- */
  const active = state.drawing && state.mode !== "idle";

  if (active) {
    if (state.mode === "eraser") {
      ensureStroke("eraser");
      appendPoint(cur);
    } else if (state.tool === "flower") {
      ensureStroke("flower");
      maybeStampFlower(cur);
    } else {
      ensureStroke("pen");
      appendPoint(cur);
    }
  } else if (state.currentStroke) {
    endStroke();
  }

  drawCursor(cur);
  updateToolPill();
}

/* ==========================================================================
   8. STATUS PILLS
   ========================================================================== */

let lastHandPill = null;
let lastToolText = "";

function updateHandPill(present) {
  if (present === lastHandPill) return;
  lastHandPill = present;
  handValue.textContent = present ? "tracked" : "none";
  pillHand.dataset.state = present ? "on" : "off";
}

function updateToolPill() {
  let text;
  let status;

  if (!state.drawing) {
    text = "paused · drawing off";
    status = "off";
  } else if (state.mode === "eraser") {
    text = "eraser · fist";
    status = "on";
  } else if (state.mode === "pen") {
    text = `${state.tool} · pointing`;
    status = "on";
  } else {
    text = "idle · open palm";
    status = "off";
  }

  if (text !== lastToolText) {
    toolValue.textContent = text;
    lastToolText = text;
  }
  pillTool.dataset.state = status;
}

/* ==========================================================================
   9. UI WIRING
   ========================================================================== */

// --- colour palette ---------------------------------------------------------
palette.addEventListener("click", (e) => {
  const btn = e.target.closest(".swatch");
  if (!btn) return;

  palette.querySelectorAll(".swatch").forEach((s) => s.classList.remove("active"));
  btn.classList.add("active");

  state.color = btn.dataset.color;
  endStroke();   // start the next stroke in the new colour, cleanly
});

// --- tool selector (only affects the POINTING gesture) ----------------------
toolSelector.addEventListener("click", (e) => {
  const btn = e.target.closest(".seg");
  if (!btn) return;

  toolSelector.querySelectorAll(".seg").forEach((s) => s.classList.remove("active"));
  btn.classList.add("active");

  state.tool = btn.dataset.tool;   // 'pen' | 'flower'
  endStroke();                     // don't connect a pen line to a flower run
  updateToolPill();
});

// --- camera selector ---------------------------------------------------------
cameraSelect.addEventListener("change", async () => {
  // Stop current stream
  const currentStream = video.srcObject;
  if (currentStream) {
    currentStream.getTracks().forEach((t) => t.stop());
  }
  // Restart with newly selected camera
  try {
    await startCamera();
    resizeCanvases();
  } catch (err) {
    console.error("Camera switch failed:", err);
  }
});

// --- drawing ON / OFF -------------------------------------------------------
drawToggle.addEventListener("click", () => {
  state.drawing = !state.drawing;
  drawToggle.classList.toggle("on", state.drawing);
  drawToggle.setAttribute("aria-checked", String(state.drawing));
  if (!state.drawing) endStroke();   // close whatever was open
  updateToolPill();
});

// --- brush size (shared by pen, eraser and flower) --------------------------
brushSlider.addEventListener("input", () => {
  state.brush = Number(brushSlider.value);
  brushValue.textContent = String(state.brush);
});

// --- clear ------------------------------------------------------------------
clearBtn.addEventListener("click", () => {
  state.strokes = [];
  state.currentStroke = null;
  state.lastStamp = null;

  const p = state.projection;
  if (p) {
    drawCtx.save();
    drawCtx.setTransform(1, 0, 0, 1, 0, 0);
    drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
    drawCtx.restore();
  }
});

/* ==========================================================================
   10. OVERLAY HELPERS
   ========================================================================== */

function showOverlay(title, text, spinner = true) {
  overlayTitle.textContent = title;
  overlayText.textContent = text;
  overlaySpinner.style.display = spinner ? "" : "none";
  overlay.classList.remove("hidden");
}

function hideOverlay() {
  overlay.classList.add("hidden");
}

function cameraErrorMessage(err) {
  const name = err && err.name;

  if (name === "NotAllowedError" || name === "PermissionDeniedError") {
    return "Camera access was denied. Enable it in your browser's site settings, then reload.";
  }
  if (name === "NotFoundError" || name === "DevicesNotFoundError") {
    return "No camera was found. Connect a webcam and reload.";
  }
  if (name === "NotReadableError" || name === "TrackStartError") {
    return "The camera is already in use by another app or tab. Close it and reload.";
  }
  if (name === "NotSupportedError" || name === "SecurityError") {
    return "This browser can't access the camera here. Try a modern browser over HTTPS or localhost.";
  }
  return "Couldn't start the camera" + (err && err.message ? ": " + err.message : ".");
}

/* ==========================================================================
   11. CAMERA + MODEL BOOTSTRAP
   ========================================================================== */

/**
 * Enumerate available cameras and populate the dropdown.
 * Iriun webcam and similar virtual cameras appear as regular video devices.
 */
async function enumerateCameras() {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoDevices = devices.filter((d) => d.kind === "videoinput");

    // Clear existing options except "Default"
    cameraSelect.innerHTML = '<option value="">Default</option>';

    for (const device of videoDevices) {
      const option = document.createElement("option");
      option.value = device.deviceId;
      // Label may be empty until permission is granted; fall back to a generic name
      option.textContent = device.label || `Camera ${videoDevices.indexOf(device) + 1}`;
      cameraSelect.appendChild(option);
    }

    // If there's only one camera, auto-select it
    if (videoDevices.length === 1) {
      cameraSelect.value = videoDevices[0].deviceId;
    }
  } catch (err) {
    console.warn("Could not enumerate cameras:", err);
  }
}

/**
 * Start the camera. When a specific deviceId is selected (e.g. Iriun webcam),
 * use it directly. Otherwise fall back to default with facingMode removed so
 * virtual webcams like Iriun are not blocked.
 */
async function startCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    const e = new Error("getUserMedia unsupported");
    e.name = "NotSupportedError";
    throw e;
  }

  const constraints = {
    video: {
      width: { ideal: 1280 },
      height: { ideal: 720 },
      // No facingMode — it can block virtual webcams like Iriun
    },
    audio: false,
  };

  // If user picked a specific camera from the dropdown, use its deviceId
  if (cameraSelect.value) {
    constraints.video.deviceId = { exact: cameraSelect.value };
  }

  const stream = await navigator.mediaDevices.getUserMedia(constraints);

  video.srcObject = stream;

  try {
    await video.play();
  } catch (err) {
    // Autoplay rejections are non-fatal for a muted MediaStream — keep going.
    if (err && err.name !== "AbortError") console.warn("video.play():", err);
  }

  // Wait until real dimensions are known so the cover math is correct.
  if (!video.videoWidth) {
    await new Promise((resolve) => {
      const done = () => resolve();
      video.addEventListener("loadedmetadata", done, { once: true });
      setTimeout(done, 1500);
    });
  }

  // Update dropdown labels now that permission has been granted
  await enumerateCameras();
}

async function createLandmarker() {
  const fileset = await FilesetResolver.forVisionTasks(WASM_URL);

  const options = (delegate) => ({
    baseOptions: {
      modelAssetPath: MODEL_URL,
      delegate,
    },
    runningMode: "VIDEO",   // required for detectForVideo()
    numHands: 1,            // single hand only
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });

  try {
    return await HandLandmarker.createFromOptions(fileset, options("GPU"));
  } catch (err) {
    console.warn("GPU delegate unavailable — falling back to CPU.", err);
    return await HandLandmarker.createFromOptions(fileset, options("CPU"));
  }
}

/* ==========================================================================
   12. BOOT
   ========================================================================== */

async function boot() {
  // Canvases must exist before anything can be projected onto them.
  resizeCanvases();

  window.addEventListener("resize", queueResize);
  window.addEventListener("orientationchange", queueResize);
  if (window.ResizeObserver) {
    new ResizeObserver(queueResize).observe(stage);
  }

  /* --- 1. camera --------------------------------------------------------- */
  showOverlay("Starting up…", "Requesting webcam access.");
  pillModel.dataset.state = "loading";
  modelValue.textContent = "loading";

  // Enumerate cameras first so the dropdown is populated
  await enumerateCameras();

  try {
    await startCamera();
  } catch (err) {
    console.error("Camera error:", err);
    showOverlay("Camera unavailable", cameraErrorMessage(err), false);
    pillModel.dataset.state = "error";
    modelValue.textContent = "error";
    return;
  }

  // Video metadata is in — recompute the projection with the true frame size.
  resizeCanvases();

  /* --- 2. hand-tracking model -------------------------------------------- */
  showOverlay("Loading hand model…", "This can take a moment on the first run.");

  try {
    landmarker = await createLandmarker();
  } catch (err) {
    console.error("HandLandmarker error:", err);
    showOverlay(
      "Hand model failed to load",
      "Check your internet connection and reload the page.",
      false
    );
    pillModel.dataset.state = "error";
    modelValue.textContent = "error";
    return;
  }

  pillModel.dataset.state = "on";
  modelValue.textContent = "ready";

  hideOverlay();
  updateToolPill();
  updateHandPill(false);

  /* --- 3. go ------------------------------------------------------------- */
  lastVideoTime = -1;
  if (rafId) cancelAnimationFrame(rafId);
  loop();
}

boot();