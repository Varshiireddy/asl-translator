import { FilesetResolver, HandLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.3/vision_bundle.mjs";

// ---------- settings ----------
const SMOOTH_FRAMES = 10;   // average predictions over this many frames
const SHOW_AT = 0.6;        // show the letter in green above this confidence
const ADD_AT = 0.8;         // add the letter to the text above this confidence...
const HOLD_FRAMES = 20;     // ...when it stays the same for this many frames
const CONNECTIONS = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],
                     [9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];

// ---------- page elements ----------
const $ = (id) => document.getElementById(id);
let video = null, canvas = null, ctx = null;

// ---------- the model: same math as the NumPy forward pass ----------
let W = null;

function dense(x, w, b, relu) {
  const out = new Float32Array(b.length);
  for (let i = 0; i < b.length; i++) {
    let s = b[i];
    const row = w[i];
    for (let j = 0; j < x.length; j++) s += row[j] * x[j];
    out[i] = relu ? Math.max(0, s) : s;
  }
  return out;
}

export function forward(x, weights = W) {
  let h = dense(x, weights.w1, weights.b1, true);
  h = dense(h, weights.w2, weights.b2, true);
  return dense(h, weights.w3, weights.b3, false);
}

export function softmax(z) {
  const m = Math.max(...z);
  const e = Array.from(z, (v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

// Same as make_features in Colab: fix aspect ratio, wrist at (0, 0), scale by hand size
export function features(landmarks, aspect) {
  const x0 = landmarks[0].x * aspect, y0 = landmarks[0].y;
  const f = [];
  let m = 0;
  for (const p of landmarks) {
    const dx = p.x * aspect - x0, dy = p.y - y0;
    f.push(dx, dy);
    m = Math.max(m, Math.abs(dx), Math.abs(dy));
  }
  return f.map((v) => v / m);
}

// ---------- app state ----------
let landmarker = null, recent = [], text = "";
let holdLetter = null, holdCount = 0, lastAdded = null, noHand = 0;
let lastVideoTime = -1, frames = 0, fpsT0 = performance.now();

function renderText() {
  $("text").textContent = text || " ";
}

function drawHand(lms) {
  ctx.lineWidth = 3;
  ctx.strokeStyle = "#4f8cff";
  for (const [a, b] of CONNECTIONS) {
    ctx.beginPath();
    ctx.moveTo(lms[a].x * canvas.width, lms[a].y * canvas.height);
    ctx.lineTo(lms[b].x * canvas.width, lms[b].y * canvas.height);
    ctx.stroke();
  }
  ctx.fillStyle = "#ff5a5a";
  for (const p of lms) {
    ctx.beginPath();
    ctx.arc(p.x * canvas.width, p.y * canvas.height, 4, 0, 2 * Math.PI);
    ctx.fill();
  }
}

function showPrediction(avg) {
  const order = [...avg.keys()].sort((a, b) => avg[b] - avg[a]);
  const k = order[0], conf = avg[k], letter = W.letters[k];

  $("letter").textContent = letter;
  $("letter").className = conf >= SHOW_AT ? "sure" : "unsure";
  $("confbar").style.width = `${Math.round(conf * 100)}%`;
  $("conf").textContent = `${Math.round(conf * 100)}%`;
  $("top3").textContent = order.slice(0, 3).map((i) => `${W.letters[i]} ${Math.round(avg[i] * 100)}%`).join("   ");

  // Hold a letter steady to type it
  if (conf >= ADD_AT && letter === holdLetter) holdCount++;
  else { holdLetter = conf >= ADD_AT ? letter : null; holdCount = 0; }
  $("hold").style.width = `${Math.min(100, (holdCount / HOLD_FRAMES) * 100)}%`;
  if (holdCount === HOLD_FRAMES && letter !== lastAdded) {
    text += letter;
    lastAdded = letter;
    renderText();
  }
}

function clearPrediction() {
  $("letter").textContent = "–";
  $("letter").className = "unsure";
  $("confbar").style.width = "0%";
  $("conf").textContent = "";
  $("top3").textContent = "Show your hand to the camera";
  $("hold").style.width = "0%";
}

function loop() {
  if (video.currentTime !== lastVideoTime && video.readyState >= 2) {
    lastVideoTime = video.currentTime;
    const result = landmarker.detectForVideo(video, performance.now());
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (result.landmarks.length) {
      const lms = result.landmarks[0];
      noHand = 0;
      drawHand(lms);
      const probs = softmax(forward(features(lms, video.videoWidth / video.videoHeight)));
      recent.push(probs);
      if (recent.length > SMOOTH_FRAMES) recent.shift();
      const avg = probs.map((_, i) => recent.reduce((s, p) => s + p[i], 0) / recent.length);
      showPrediction(avg);
    } else {
      recent = [];
      holdLetter = null; holdCount = 0;
      if (++noHand > 15) lastAdded = null;   // hand gone: allow the same letter again
      clearPrediction();
    }

    frames++;
    const now = performance.now();
    if (now - fpsT0 > 1000) {
      $("fps").textContent = `${Math.round((frames * 1000) / (now - fpsT0))} fps`;
      frames = 0; fpsT0 = now;
    }
  }
  requestAnimationFrame(loop);
}

async function start() {
  video = $("video");
  canvas = $("overlay");
  ctx = canvas.getContext("2d");
  $("start").disabled = true;
  $("status").textContent = "Loading models…";
  try {
    W = await (await fetch("weights.json")).json();
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.3/wasm");
    const options = (delegate) => ({
      baseOptions: {
        modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
        delegate,
      },
      runningMode: "VIDEO",
      numHands: 1,
      minHandDetectionConfidence: 0.3,
    });
    try {
      landmarker = await HandLandmarker.createFromOptions(vision, options("GPU"));
    } catch {
      landmarker = await HandLandmarker.createFromOptions(vision, options("CPU"));
    }

    $("status").textContent = "Starting camera…";
    video.srcObject = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 }, audio: false });
    await video.play();
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    $("start").hidden = true;
    $("status").textContent = "Running in your browser. Your video never leaves this device.";
    clearPrediction();
    requestAnimationFrame(loop);
  } catch (err) {
    console.error(err);
    $("status").textContent = `Couldn't start: ${err.message}. Allow camera access and reload.`;
    $("start").disabled = false;
  }
}

if (typeof document !== "undefined" && $("start")) {
  $("start").addEventListener("click", start);
  $("space").addEventListener("click", () => { text += " "; lastAdded = null; renderText(); });
  $("back").addEventListener("click", () => { text = text.slice(0, -1); lastAdded = null; renderText(); });
  $("clear").addEventListener("click", () => { text = ""; lastAdded = null; renderText(); });
  renderText();
  if (location.hostname.endsWith("github.io")) {
    $("repo").href = `https://github.com/${location.hostname.split(".")[0]}/${location.pathname.split("/")[1]}`;
  }
}
