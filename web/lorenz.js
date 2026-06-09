// Syne — interactive Lorenz viewer.
//
// Consumes the per-frame control JSON exported by `syne render --controls`
// (the same semantic -> Lorenz mapping the offline renderer uses) and animates
// a morphing attractor in real time, synced to the audio's playback clock.
//
// The Python side owns the *meaning* (tags -> rho/sigma/beta/speed/...); this
// file is purely an integrator + renderer that reads those connectors.

import * as THREE from "three";

const TAIL = 2600;            // trajectory points kept in the comet
const SUBSTEPS = 6;          // integration substeps per animation frame
const Z_CENTER = 25.0;

let control = null;          // loaded control JSON
let audioEl = document.getElementById("audio");
let playing = false;
let synthClock = 0;          // fallback time when no audio is loaded

// ---- three.js scene -------------------------------------------------------
const canvas = document.getElementById("gl");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x05060f, 0.012);

const camera = new THREE.PerspectiveCamera(55, 1, 0.1, 1000);
camera.position.set(0, 0, 95);

// the comet: a line whose vertex colors fade along the tail
const positions = new Float32Array(TAIL * 3);
const colors = new Float32Array(TAIL * 3);
const geom = new THREE.BufferGeometry();
geom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
geom.setAttribute("color", new THREE.BufferAttribute(colors, 3));
const mat = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true,
  blending: THREE.AdditiveBlending, depthWrite: false });
const line = new THREE.Line(geom, mat);
scene.add(line);

// a glowing sprite for the comet head
const head = new THREE.Mesh(
  new THREE.SphereGeometry(0.7, 16, 16),
  new THREE.MeshBasicMaterial({ color: 0xffffff })
);
scene.add(head);

const pivot = new THREE.Group();
pivot.add(line); pivot.add(head);
scene.remove(line); scene.add(pivot);

function resize() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h; camera.updateProjectionMatrix();
}
addEventListener("resize", resize); resize();

// ---- Lorenz integrator ----------------------------------------------------
let state = { x: 0.1, y: 0.0, z: 0.0 };
const trail = [];            // {x,y,z,h,g}
let lastSeedIdx = 0;
let rotation = 0;

function deriv(s, sig, rho, beta) {
  return {
    x: sig * (s.y - s.x),
    y: s.x * (rho - s.z) - s.y,
    z: s.x * s.y - beta * s.z,
  };
}
function rk4(s, sig, rho, beta, dt) {
  const k1 = deriv(s, sig, rho, beta);
  const s2 = { x: s.x + 0.5 * dt * k1.x, y: s.y + 0.5 * dt * k1.y, z: s.z + 0.5 * dt * k1.z };
  const k2 = deriv(s2, sig, rho, beta);
  const s3 = { x: s.x + 0.5 * dt * k2.x, y: s.y + 0.5 * dt * k2.y, z: s.z + 0.5 * dt * k2.z };
  const k3 = deriv(s3, sig, rho, beta);
  const s4 = { x: s.x + dt * k3.x, y: s.y + dt * k3.y, z: s.z + dt * k3.z };
  const k4 = deriv(s4, sig, rho, beta);
  return {
    x: s.x + (dt / 6) * (k1.x + 2 * k2.x + 2 * k3.x + k4.x),
    y: s.y + (dt / 6) * (k1.y + 2 * k2.y + 2 * k3.y + k4.y),
    z: s.z + (dt / 6) * (k1.z + 2 * k2.z + 2 * k3.z + k4.z),
  };
}

// sample a per-frame channel at time t (seconds) via linear interpolation
function sample(arr, t) {
  const times = control.times;
  if (t <= times[0]) return arr[0];
  if (t >= times[times.length - 1]) return arr[arr.length - 1];
  // times are roughly uniform; binary search keeps it exact
  let lo = 0, hi = times.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (times[mid] <= t) lo = mid; else hi = mid;
  }
  const f = (t - times[lo]) / (times[hi] - times[lo] || 1);
  return arr[lo] + f * (arr[hi] - arr[lo]);
}

function hsvToRgb(h, s, v) {
  const i = Math.floor(h * 6), f = h * 6 - i;
  const p = v * (1 - s), q = v * (1 - f * s), t = v * (1 - (1 - f) * s);
  const m = [[v, t, p], [q, v, p], [p, v, t], [p, q, v], [t, p, v], [v, p, q]][i % 6];
  return m;
}

function currentTime() {
  if (audioEl.src && !audioEl.paused) return audioEl.currentTime;
  if (audioEl.src) return audioEl.currentTime;
  return synthClock;
}

function step(dtReal) {
  if (!control) return;
  const t = currentTime();

  // section re-seed: nudge the state at structural boundaries
  while (lastSeedIdx < control.seeds.length && control.seeds[lastSeedIdx] <= t) {
    state.x += (Math.random() - 0.5) * 1.2;
    state.y += (Math.random() - 0.5) * 1.2;
    state.z += (Math.random() - 0.5) * 1.2;
    lastSeedIdx++;
  }

  const rho = sample(control.rho, t);
  const sig = sample(control.sigma, t);
  const beta = sample(control.beta, t);
  const speed = sample(control.speed, t);
  const kick = sample(control.kick, t);
  const jitter = sample(control.jitter, t);
  const glow = sample(control.glow, t);
  const hue = sample(control.hue, t);

  const dt = (control.base_dt * Math.max(speed, 0.05)) / SUBSTEPS;
  for (let i = 0; i < SUBSTEPS; i++) {
    state = rk4(state, sig, rho, beta, dt);
    if (jitter > 0) {
      state.x += (Math.random() - 0.5) * jitter * 0.24;
      state.y += (Math.random() - 0.5) * jitter * 0.24;
      state.z += (Math.random() - 0.5) * jitter * 0.24;
    }
    trail.push({ x: state.x, y: state.y, z: state.z - Z_CENTER, h: hue, g: glow });
  }
  if (kick > 0.2) {
    state.x += (Math.random() - 0.5) * kick * 1.4;
    state.y += (Math.random() - 0.5) * kick * 1.4;
  }
  while (trail.length > TAIL) trail.shift();

  // upload trail to geometry, fading + coloring along the tail
  const sat = control.saturation;
  const n = trail.length;
  for (let i = 0; i < n; i++) {
    const p = trail[i];
    positions[i * 3] = p.x; positions[i * 3 + 1] = p.z; positions[i * 3 + 2] = p.y;
    const fade = Math.pow((i + 1) / n, 1.5);
    const v = fade * (0.3 + 0.8 * p.g);
    const rgb = hsvToRgb(p.h % 1, sat, 1);
    colors[i * 3] = rgb[0] * v; colors[i * 3 + 1] = rgb[1] * v; colors[i * 3 + 2] = rgb[2] * v;
  }
  geom.setDrawRange(0, n);
  geom.attributes.position.needsUpdate = true;
  geom.attributes.color.needsUpdate = true;
  if (n > 0) {
    const p = trail[n - 1];
    head.position.set(p.x, p.z, p.y);
    head.material.color.setRGB(...hsvToRgb(p.h % 1, sat * 0.3, 1));
  }

  rotation += control.rotation_speed * 0.35 * dtReal;
  pivot.rotation.y = rotation;
}

// ---- loop -----------------------------------------------------------------
let last = performance.now();
function loop(now) {
  const dtReal = Math.min((now - last) / 1000, 0.05); last = now;
  if (playing && !audioEl.src) synthClock += dtReal;
  step(dtReal);
  renderer.render(scene, camera);
  requestAnimationFrame(loop);
}
requestAnimationFrame(loop);

// ---- UI -------------------------------------------------------------------
const hud = document.getElementById("hud");
function resetState() {
  state = { x: 0.1, y: 0, z: 0 }; trail.length = 0; lastSeedIdx = 0;
  synthClock = 0; rotation = 0;
}

document.getElementById("ctrlFile").addEventListener("change", async (e) => {
  const file = e.target.files[0]; if (!file) return;
  control = JSON.parse(await file.text());
  resetState();
  hud.textContent =
    `loaded ${control.times.length} frames\n` +
    `rho ${control.rho[0].toFixed(1)} · sigma ${control.sigma[0].toFixed(1)} ` +
    `· beta ${control.beta[0].toFixed(2)}\n` +
    `${control.seeds.length} section re-seeds`;
});

document.getElementById("audioFile").addEventListener("change", (e) => {
  const file = e.target.files[0]; if (!file) return;
  audioEl.src = URL.createObjectURL(file);
  resetState();
});

document.getElementById("play").addEventListener("click", () => {
  if (audioEl.src) { audioEl.paused ? audioEl.play() : audioEl.pause(); }
  else { playing = !playing; }
});
document.getElementById("reset").addEventListener("click", () => {
  resetState(); if (audioEl.src) audioEl.currentTime = 0;
});
audioEl.addEventListener("seeked", resetState);
