// Quantum Defense — MVP interactivo: Mundo 2 (Meseta Radial) y Mundo 3 (Cumbres del Vórtice)
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { contourSegments, bilinear, accuracyOnGrid, strokeClassifier, liveStress, lerp } from './physics.js';

// ------------------------------------------------------------------ datos y estado
const WORLD_LORE = {
  2: {
    title: 'La Meseta Radial',
    lore: 'Las Llamitas de Coherencia se enroscan en un anillo interior; los Slimes de Entropía las rodean. Ninguna línea recta puede separarlas… hasta que elevas el terreno. Un Slime infiltrado se esconde entre las llamitas: no lo memorices.',
    hint: 'Traza una línea (fallará) → luego «Elevar» y baja el plano entre los anillos.',
    hasLift: true,
  },
  3: {
    title: 'Las Cumbres del Vórtice',
    lore: 'Dos lunas entrelazadas. Aquí el terreno lo deforma un kernel cuántico: el ZZFeatureMap codifica cada punto en un estado y la fidelidad |⟨φ(x)|φ(y)⟩|² define la distancia. Más reps = más expresividad… y más peligro de memorizar al infiltrado.',
    hint: 'Prueba reps=3 con C=1000 y observa cómo se quiebra el cristal.',
    hasLift: false,
  },
};
const KERNEL_FORMULA = {
  linear: 'k(x,y) = ⟨x, y⟩',
  poly: 'k(x,y) = (⟨x,y⟩ + 1)<sup>d</sup>',
  rbf: 'k(x,y) = exp(−γ‖x−y‖²)',
  zz: 'k(x,y) = |⟨0|U<sup>†</sup>(y) U(x)|0⟩|²  <i>(FidelityQuantumKernel + ZZFeatureMap)</i>',
};
const SIZE = 10, H_SCALE = 1.6, N = 80;
const $ = (id) => document.getElementById(id);
const store = JSON.parse(localStorage.getItem('qd-progress') || '{}');

const S = {
  world: 0, data: null, cfgIdx: -1, cfg: null,
  hFrom: new Float32Array(N * N), hTo: new Float32Array(N * N), hCur: new Float32Array(N * N), morphT: 1,
  stressT: 0, stressTarget: 0, shattered: false, shatterT: -1, formT: 1,
  nisq: 0, waveT: -1, strokeMode: false, stroke: [], showTest: false,
  lift: 0, plane: 1.3, liveStressBonus: 0,
};
const cache = {};

// ------------------------------------------------------------------ escena
const canvas = $('stage');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x070920, 0.028);
const camera = new THREE.PerspectiveCamera(48, 1, 0.1, 200);
camera.position.set(10.5, 9, 12);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true; controls.enablePan = false;
controls.maxPolarAngle = Math.PI * 0.48; controls.minDistance = 8; controls.maxDistance = 30;

scene.add(new THREE.AmbientLight(0x8890ff, 0.55));
const key = new THREE.DirectionalLight(0xffffff, 1.1); key.position.set(6, 12, 4); scene.add(key);
const rim = new THREE.PointLight(0xa46bff, 40, 40); rim.position.set(-8, 5, -6); scene.add(rim);
const cyanL = new THREE.PointLight(0x4be8ff, 30, 40); cyanL.position.set(8, 4, 8); scene.add(cyanL);

// campo de estrellas
{
  const g = new THREE.BufferGeometry(); const p = [];
  for (let i = 0; i < 700; i++) {
    const r = 40 + Math.random() * 50, a = Math.random() * 6.283, b = Math.random() * 1.2 + 0.1;
    p.push(Math.cos(a) * r, Math.sin(b) * r, Math.sin(a) * r);
  }
  g.setAttribute('position', new THREE.Float32BufferAttribute(p, 3));
  scene.add(new THREE.Points(g, new THREE.PointsMaterial({ color: 0xaab4ff, size: 0.35, transparent: true, opacity: 0.8 })));
}

// terreno
const tGeo = new THREE.PlaneGeometry(SIZE, SIZE, N - 1, N - 1); tGeo.rotateX(-Math.PI / 2);
tGeo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(N * N * 3), 3));
const tMat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.55, metalness: 0.25, side: THREE.DoubleSide, transparent: true });
const terrain = new THREE.Mesh(tGeo, tMat); scene.add(terrain);
const wire = new THREE.Mesh(tGeo, new THREE.MeshBasicMaterial({ color: 0x7fa0ff, wireframe: true, transparent: true, opacity: 0.12 }));
scene.add(wire);
const base = new THREE.Mesh(new THREE.CylinderGeometry(SIZE * 0.78, SIZE * 0.82, 0.3, 64, 1, true),
  new THREE.MeshBasicMaterial({ color: 0x3a46aa, transparent: true, opacity: 0.18, side: THREE.DoubleSide }));
base.position.y = -2.6; scene.add(base);

// puntos de juego
const pointsGroup = new THREE.Group(); scene.add(pointsGroup);
let trainMeshes = [], testMeshes = [];
const llamaGeo = new THREE.IcosahedronGeometry(0.2, 1), slimeGeo = new THREE.SphereGeometry(0.22, 20, 14);
const llamaMat = new THREE.MeshStandardMaterial({ color: 0x4be8ff, emissive: 0x1ab7d6, emissiveIntensity: 1.2, roughness: 0.2 });
const slimeMat = new THREE.MeshStandardMaterial({ color: 0xa6ff5a, emissive: 0x4a9a1c, emissiveIntensity: 0.9, roughness: 0.35, transparent: true, opacity: 0.92 });
const ghostMat = (c) => new THREE.MeshBasicMaterial({ color: c, transparent: true, opacity: 0.55, wireframe: true });

// escudo (cristal) — triángulos independientes que pueden hacerse añicos
const shield = { mesh: null, shards: [], rest: null, ready: false };
const shieldMat = new THREE.MeshPhysicalMaterial({
  color: 0x4be8ff, emissive: 0x2a7a99, emissiveIntensity: 0.9, transparent: true, opacity: 0.55,
  roughness: 0.05, metalness: 0.1, transmission: 0.0, side: THREE.DoubleSide, depthWrite: false,
});
const shieldEdge = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0xbff6ff, transparent: true, opacity: 0.9 }));
scene.add(shieldEdge);

// trazo del jugador
const strokeLine = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0xffd36b }));
const strokeWall = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshBasicMaterial({ color: 0xffd36b, transparent: true, opacity: 0.22, side: THREE.DoubleSide, depthWrite: false }));
scene.add(strokeLine, strokeWall);
const planeMesh = new THREE.Mesh(new THREE.CircleGeometry(SIZE * 0.55, 64),
  new THREE.MeshBasicMaterial({ color: 0xffd36b, transparent: true, opacity: 0.25, side: THREE.DoubleSide, depthWrite: false }));
planeMesh.rotation.x = -Math.PI / 2; planeMesh.visible = false; scene.add(planeMesh);

// partículas de escombros
const debris = { mesh: null };

// ------------------------------------------------------------------ utilidades de mapeo
const toW = (v) => ((v - S.data.grid.min) / (S.data.grid.max - S.data.grid.min) - 0.5) * SIZE;
const fromW = (w) => (w / SIZE + 0.5) * (S.data.grid.max - S.data.grid.min) + S.data.grid.min;
const softH = (d) => H_SCALE * Math.tanh(d * 0.6);
const terrainH = (x, z) => bilinear(S.hCur, N, (x / SIZE) + 0.5, (z / SIZE) + 0.5);

function upsample(dec, G) {
  const out = new Float32Array(N * N);
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
    const gi = (i / (N - 1)) * (G - 1), gj = (j / (N - 1)) * (G - 1);
    const i0 = Math.min(Math.floor(gi), G - 2), j0 = Math.min(Math.floor(gj), G - 2);
    const fi = gi - i0, fj = gj - j0;
    const v = lerp(lerp(dec[i0][j0], dec[i0][j0 + 1], fj), lerp(dec[i0 + 1][j0], dec[i0 + 1][j0 + 1], fj), fi);
    out[i * N + j] = softH(v);
  }
  return out;
}

// ------------------------------------------------------------------ carga de datos
async function loadWorld(w) {
  if (!cache[w]) cache[w] = await (await fetch(`data/world${w}.json`)).json();
  return cache[w];
}

async function selectWorld(w) {
  S.data = await loadWorld(w); S.world = w; S.cfgIdx = -1; S.cfg = null; S.shattered = false;
  document.querySelectorAll('.world-tab').forEach((b) => b.classList.toggle('active', +b.dataset.world === w));
  $('lore').textContent = WORLD_LORE[w].lore;
  $('hint').textContent = WORLD_LORE[w].hint;
  $('lift-box').hidden = !WORLD_LORE[w].hasLift;
  S.lift = 0; $('lift').value = 0;
  clearStroke(); buildPoints(); buildSpellbook(); clearShield();
  S.hFrom.fill(0); S.hTo.fill(0); S.hCur.fill(0); S.morphT = 1; applyTerrain();
  resetHud();
  castSpell(S.data.configs.findIndex((c) => c.id === S.data.recommended) >= 0 ? 0 : 0, true);
}

function buildPoints() {
  [...trainMeshes, ...testMeshes].forEach((m) => pointsGroup.remove(m));
  trainMeshes = []; testMeshes = [];
  const { train, test } = S.data;
  train.X.forEach((p, i) => {
    const slime = train.y[i] < 0;
    const m = new THREE.Mesh(slime ? slimeGeo : llamaGeo, slime ? slimeMat : llamaMat);
    m.userData = { p, slime, infiltrator: train.outlier[i], phase: Math.random() * 6.28 };
    if (train.outlier[i]) {
      // Slime infiltrado: aura de peligro (etiqueta corrupta)
      m.userData.trueSlime = !slime; // es del bando contrario al que dice su etiqueta
      const ring = new THREE.Mesh(new THREE.TorusGeometry(0.42, 0.03, 8, 40), new THREE.MeshBasicMaterial({ color: 0xff4d6d }));
      ring.rotation.x = Math.PI / 2; m.add(ring); m.userData.ring = ring;
    }
    pointsGroup.add(m); trainMeshes.push(m);
  });
  test.X.forEach((p, i) => {
    const m = new THREE.Mesh(new THREE.OctahedronGeometry(0.13), ghostMat(test.y[i] > 0 ? 0x4be8ff : 0xa6ff5a));
    m.userData = { p, y: test.y[i] }; m.visible = S.showTest; pointsGroup.add(m); testMeshes.push(m);
  });
}

function placePoints(t) {
  const lift = S.lift;
  const place = (m, p, hover, trainFlag) => {
    const x = toW(p[0]), z = toW(p[1]);
    const r2 = (p[0] - Math.PI / 2) ** 2 + (p[1] - Math.PI / 2) ** 2;
    const th = terrainH(x, z) * (1 - lift) + lift * 0.5 * r2 * (S.world === 2 ? 1 : 0);
    const y = (S.world === 2 ? th : terrainH(x, z)) + hover + Math.sin(t * 2 + (m.userData.phase || 0)) * 0.04;
    m.position.set(x, y, z);
  };
  trainMeshes.forEach((m) => {
    place(m, m.userData.p, 0.3, true);
    if (m.userData.slime) { const k = 1 + Math.sin(t * 3 + m.userData.phase) * 0.08; m.scale.set(1.1 * k, 0.8 / k, 1.1 * k); }
    else { m.rotation.y = t * 1.2; m.rotation.x = t * 0.7; }
    if (m.userData.ring) { m.userData.ring.rotation.z = t * 2; m.userData.ring.scale.setScalar(1 + 0.15 * Math.sin(t * 5)); }
  });
  testMeshes.forEach((m) => { if (m.visible) { place(m, m.userData.p, 0.22, false); m.rotation.y = t; } });
  // plano de elevación
  planeMesh.visible = S.world === 2 && S.lift > 0.02;
  planeMesh.position.y = S.lift * 0.5 * S.plane;
  tMat.opacity = 1 - 0.7 * (S.world === 2 ? S.lift : 0);
  wire.visible = S.lift < 0.7;
}

// ------------------------------------------------------------------ terreno
function applyTerrain() {
  const pos = tGeo.attributes.position, col = tGeo.attributes.color;
  for (let k = 0; k < N * N; k++) {
    const h = S.hCur[k]; pos.setY(k, h);
    const t = Math.max(-1, Math.min(1, h / H_SCALE)); // [-1,1]
    const a = Math.abs(t);
    if (t >= 0) col.setXYZ(k, lerp(0.1, 0.1, a), lerp(0.14, 0.78, a), lerp(0.3, 0.95, a));
    else col.setXYZ(k, lerp(0.1, 0.55, a), lerp(0.1, 0.12, a), lerp(0.28, 0.65, a));
  }
  pos.needsUpdate = true; col.needsUpdate = true; tGeo.computeVertexNormals();
}

// ------------------------------------------------------------------ escudo
function clearShield() {
  if (shield.mesh) { scene.remove(shield.mesh); shield.mesh.geometry.dispose(); shield.mesh = null; }
  shield.shards = []; shield.ready = false; shieldEdge.geometry.setAttribute('position', new THREE.Float32BufferAttribute([], 3));
  if (debris.mesh) { scene.remove(debris.mesh); debris.mesh = null; }
}

function buildShield() {
  clearShield();
  const { size: G, min, max } = S.data.grid;
  const segs = contourSegments(S.cfg.decision, G, min, max, (v) => toW(v));
  const y0 = -0.1, y1 = 1.5;
  const rest = []; const shards = [];
  for (const [ax, az, bx, bz] of segs) {
    const tris = [
      [[ax, y0, az], [bx, y0, bz], [bx, y1, bz]],
      [[ax, y0, az], [bx, y1, bz], [ax, y1, az]],
    ];
    for (const t of tris) {
      const c = new THREE.Vector3(); t.forEach((v) => c.add(new THREE.Vector3(...v))); c.multiplyScalar(1 / 3);
      shards.push({
        rest: t.map((v) => new THREE.Vector3(...v).sub(c)), c,
        pos: c.clone(), vel: new THREE.Vector3(), q: new THREE.Quaternion(),
        axis: new THREE.Vector3(Math.random() - 0.5, Math.random() - 0.5, Math.random() - 0.5).normalize(),
        spin: (Math.random() - 0.5) * 12, delay: Math.random() * 0.12, seed: Math.random() * 100,
      });
    }
  }
  shield.shards = shards;
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(shards.length * 9), 3));
  shield.mesh = new THREE.Mesh(geo, shieldMat); shield.mesh.frustumCulled = false; scene.add(shield.mesh);
  shield.ready = true; shieldEdge.frustumCulled = false;
  S.shattered = false; S.shatterT = -1; S.formT = 0; S.stressT = 0;
  shieldMat.opacity = 0.55;
}

const _v = new THREE.Vector3(), _q = new THREE.Quaternion();
function updateShield(dt, t) {
  if (!shield.ready) return;
  const sh = shield.shards, pos = shield.mesh.geometry.attributes.position;
  const stress = Math.min(S.stressT, 1.2);
  const grow = Math.min(1, S.formT); S.formT = Math.min(1.2, S.formT + dt / 0.9);
  const ease = 1 - Math.pow(1 - grow, 3);
  const edge = [];
  for (let i = 0; i < sh.length; i++) {
    const s = sh[i];
    if (S.shatterT >= 0) {
      const ts = S.shatterT - s.delay;
      if (ts > 0) {
        s.vel.y -= 14 * dt; s.pos.addScaledVector(s.vel, dt);
        _q.setFromAxisAngle(s.axis, s.spin * dt); s.q.premultiply(_q);
        const fl = terrainH(s.pos.x, s.pos.z) + 0.05;
        if (s.pos.y < fl) { s.pos.y = fl; s.vel.y *= -0.3; s.vel.x *= 0.7; s.vel.z *= 0.7; s.spin *= 0.8; }
      }
    } else {
      // vibración: el cristal "canta" al tensarse
      const amp = Math.max(0, stress - 0.35) * 0.05 + S.nisq * 0.05;
      s.pos.set(s.c.x + Math.sin(t * 40 + s.seed) * amp, s.c.y * ease + Math.sin(t * 33 + s.seed * 2) * amp * 0.6, s.c.z + Math.cos(t * 37 + s.seed) * amp);
    }
    for (let k = 0; k < 3; k++) {
      _v.copy(s.rest[k]); if (S.shatterT < 0) _v.y *= ease; _v.applyQuaternion(s.q).add(s.pos);
      if (S.shatterT < 0) _v.y += (s.pos.y - s.c.y * ease) * 0; // sin efecto; mantiene la forma
      pos.setXYZ(i * 3 + k, _v.x, _v.y, _v.z);
    }
    if (S.shatterT < 0 && i % 2 === 0) {
      const a = pos, b = i * 3;
      edge.push(a.getX(b), a.getY(b), a.getZ(b), a.getX(b + 2), a.getY(b + 2), a.getZ(b + 2));
    }
  }
  pos.needsUpdate = true;
  shieldEdge.geometry.setAttribute('position', new THREE.Float32BufferAttribute(edge, 3));
  shieldEdge.visible = S.shatterT < 0;

  // color por esfuerzo: cian → ámbar → rojo; el cristal pierde transparencia al agrietarse
  const sc = Math.min(stress, 1);
  const col = new THREE.Color(0x4be8ff).lerp(new THREE.Color(0xffd36b), Math.min(1, sc * 1.6)).lerp(new THREE.Color(0xff4d6d), Math.max(0, sc - 0.6) / 0.4);
  shieldMat.color.copy(col); shieldMat.emissive.copy(col).multiplyScalar(0.45);
  shieldEdge.material.color.copy(col).lerp(new THREE.Color(0xffffff), 0.5);
  if (S.shatterT >= 0) {
    S.shatterT += dt; shieldMat.opacity = Math.max(0, 0.7 - Math.max(0, S.shatterT - 2.2) * 0.5);
  }
}

function shatter() {
  if (S.shatterT >= 0) return;
  S.shatterT = 0; S.shattered = true;
  const infil = trainMeshes.find((m) => m.userData.infiltrator);
  const o = infil ? infil.position : new THREE.Vector3();
  for (const s of shield.shards) {
    const dir = s.pos.clone().sub(o); dir.y = 0; if (dir.lengthSq() < 1e-4) dir.set(Math.random() - 0.5, 0, Math.random() - 0.5);
    dir.normalize();
    const f = 1.5 + Math.random() * 3.2;
    s.vel.set(dir.x * f + (Math.random() - 0.5) * 2, 2 + Math.random() * 5, dir.z * f + (Math.random() - 0.5) * 2);
  }
  canvas.classList.add('shake'); setTimeout(() => canvas.classList.remove('shake'), 520);
  glassSound(); toast('💥 ¡El escudo se quebró! Sobreajustaste al Slime infiltrado.', 'bad');
  setStressText('bad', 'Cristal destruido: el kernel memorizó ruido y no generaliza.');
}

function glassSound() {
  try {
    const ac = new (window.AudioContext || window.webkitAudioContext)();
    const len = ac.sampleRate * 0.9, buf = ac.createBuffer(1, len, ac.sampleRate), d = buf.getChannelData(0);
    for (let i = 0; i < len; i++) {
      const env = Math.exp(-i / (ac.sampleRate * 0.18)) + 0.5 * Math.exp(-((i % 4000) / 900)) * Math.exp(-i / (ac.sampleRate * 0.5));
      d[i] = (Math.random() * 2 - 1) * env;
    }
    const src = ac.createBufferSource(), hp = ac.createBiquadFilter(), g = ac.createGain();
    hp.type = 'highpass'; hp.frequency.value = 2500; g.gain.value = 0.35;
    src.buffer = buf; src.connect(hp).connect(g).connect(ac.destination); src.start();
  } catch (e) { /* audio opcional */ }
}

// ------------------------------------------------------------------ hechizos
function buildSpellbook() {
  const ul = $('spell-list'); ul.innerHTML = '';
  S.data.configs.forEach((c, i) => {
    const li = document.createElement('li'); li.className = 'spell'; li.id = `spell-${c.id}`;
    const rec = store[`${S.world}:${c.id}`];
    const stars = rec ? (rec.shattered ? '<span class="st broke">✖ roto</span>' : `<span class="st">${'★'.repeat(rec.stars)}${'☆'.repeat(3 - rec.stars)}</span>`) : '<span class="st">???</span>';
    li.innerHTML = `<div class="row"><span class="name">${c.label}</span><span class="badge ${c.kernel}">${c.kernel === 'zz' ? 'cuántico' : c.kernel}</span></div>
      <div class="row"><span class="sub">C=${c.C} · complejidad ${c.complexity}</span>${stars}</div>`;
    li.addEventListener('click', () => castSpell(i));
    ul.appendChild(li);
  });
  updateTotalStars();
}
function updateTotalStars() {
  let t = 0; for (const k in store) if (!store[k].shattered) t += store[k].stars;
  $('stars-total').textContent = `★ ${t}`;
}

function castSpell(i, silent) {
  const c = S.data.configs[i]; if (!c) return;
  S.cfgIdx = i; S.cfg = c; S.done = false;
  document.querySelectorAll('.spell').forEach((el) => el.classList.toggle('active', el.id === `spell-${c.id}`));
  S.hFrom.set(S.hCur); S.hTo = upsample(c.decision, S.data.grid.size); S.morphT = 0;
  buildShield();
  S.stressTarget = Math.max(0, c.stress); S.liveStressBonus = 0; S.nisq = 0;
  const d = $('spell-detail');
  d.innerHTML = `<b>${c.label}</b><br><code>${KERNEL_FORMULA[c.kernel]}</code><br>
    Vectores soporte: <b>${c.n_support}</b> · rugosidad <b>${c.roughness}</b><br>
    ${c.kernel === 'zz' ? `ZZFeatureMap con <b>reps=${c.params.reps}</b> (profundidad del circuito).` : ''}
    ${c.complexity > 2 ? '<br><span style="color:#ffd36b">⚠ Alta complejidad: riesgo de memorizar ruido.</span>' : ''}`;
  fillStats(c);
  setStressText('', 'El vórtice presiona el escudo…');
  updateStrokeReport();
}

function fillStats(c) {
  const pc = (v) => `${Math.round(v * 100)}%`;
  $('s-train').textContent = pc(c.train_acc); $('s-test').textContent = pc(c.test_acc);
  $('s-robust').textContent = pc(c.robust_acc); $('s-gap').textContent = (c.gap * 100).toFixed(0) + 'pp';
  $('s-margin').textContent = c.margin.toFixed(2);
  $('s-capt').textContent = S.data.train.outlier.some(Boolean) ? (c.captured ? 'CAPTURADO' : 'ignorado') : '—';
  $('s-capt').style.color = c.captured ? '#ff4d6d' : '#a6ff5a';
}
function resetHud() {
  ['s-train', 's-test', 's-robust', 's-gap', 's-margin', 's-capt'].forEach((id) => ($(id).textContent = '—'));
  $('stress-fill').style.width = '0'; $('stress-val').textContent = '0%'; $('stars').textContent = '☆☆☆';
}
function setStressText(cls, txt) { const e = $('stress-state'); e.className = `stress-state ${cls}`; e.textContent = txt; }

function finalizeSpell() {
  const c = S.cfg; if (!c) return;
  const broke = S.shattered;
  const stars = broke ? 0 : c.stars;
  $('stars').textContent = '★'.repeat(stars) + '☆'.repeat(3 - stars);
  store[`${S.world}:${c.id}`] = { stars: c.stars, shattered: c.shattered };
  localStorage.setItem('qd-progress', JSON.stringify(store));
  const li = $(`spell-${c.id}`);
  if (li) {
    const st = li.querySelector('.st');
    st.className = c.shattered ? 'st broke' : 'st';
    st.textContent = c.shattered ? '✖ roto' : '★'.repeat(c.stars) + '☆'.repeat(3 - c.stars);
    li.classList.toggle('shattered', c.shattered);
  }
  updateTotalStars();
  if (!broke) {
    const s = c.stress;
    setStressText(s > 0.7 ? 'warn' : 'ok', s > 0.7
      ? 'El cristal resiste, pero está al límite: cuidado con la complejidad.'
      : 'Escudo estable: generaliza más allá del entrenamiento.');
    if (c.stars === 3) toast('✨ ¡Escudo perfecto! Simple, preciso y robusto.', 'good');
  }
}

// ------------------------------------------------------------------ Onda NISQ en vivo
function nisqWave() {
  if (!S.cfg || S.shatterT >= 0) return;
  const rules = S.data.overfit_rules;
  const { X, y } = S.data.test;
  const g = S.data.grid, G = g.size;
  const base = accuracyOnGrid(S.cfg.decision, G, g.min, g.max, X, y);
  let acc = 0; const draws = 6;
  for (let d = 0; d < draws; d++) {
    const Xj = X.map((p) => p.map((v) => Math.min(Math.PI, Math.max(0, v + gauss() * rules.jitter_sigma * 1.4))));
    acc += accuracyOnGrid(S.cfg.decision, G, g.min, g.max, Xj, y);
  }
  acc /= draws;
  const drop = Math.max(0, base - acc);
  S.nisq = 1; S.waveT = 0;
  S.stressTarget = liveStress(S.cfg, rules, drop);
  $('s-robust').textContent = `${Math.round(acc * 100)}%`;
  toast(`🌊 Onda NISQ: precisión ${Math.round(base * 100)}% → ${Math.round(acc * 100)}%`, drop > 0.08 ? 'bad' : 'good');
  S.stressT = Math.min(S.stressT, S.stressTarget);
}
function gauss() { return Math.sqrt(-2 * Math.log(Math.random() || 1e-9)) * Math.cos(6.2832 * Math.random()); }

// ------------------------------------------------------------------ trazo del jugador
const ray = new THREE.Raycaster(), ndc = new THREE.Vector2(), ground = new THREE.Plane(new THREE.Vector3(0, 1, 0), -0.2);
let drawing = false;
function groundPoint(ev) {
  const r = canvas.getBoundingClientRect();
  ndc.set(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
  ray.setFromCamera(ndc, camera);
  const p = new THREE.Vector3(); return ray.ray.intersectPlane(ground, p) ? p : null;
}
canvas.addEventListener('pointerdown', (e) => {
  if (!S.strokeMode) return;
  drawing = true; S.stroke = []; canvas.setPointerCapture(e.pointerId); addStrokePoint(e);
});
canvas.addEventListener('pointermove', (e) => { if (drawing) addStrokePoint(e); });
canvas.addEventListener('pointerup', () => { if (drawing) { drawing = false; updateStrokeReport(true); } });
function addStrokePoint(e) {
  const p = groundPoint(e); if (!p) return;
  const lim = SIZE / 2; p.x = Math.max(-lim, Math.min(lim, p.x)); p.z = Math.max(-lim, Math.min(lim, p.z));
  const last = S.stroke[S.stroke.length - 1];
  if (!last || Math.hypot(last.x - p.x, last.z - p.z) > 0.12) { S.stroke.push(p); renderStroke(); }
}
function renderStroke() {
  const pts = S.stroke; strokeLine.geometry.setFromPoints(pts.map((p) => new THREE.Vector3(p.x, 0.25, p.z)));
  const v = [];
  for (let i = 0; i < pts.length - 1; i++) {
    const a = pts[i], b = pts[i + 1];
    v.push(a.x, 0, a.z, b.x, 0, b.z, b.x, 1.4, b.z, a.x, 0, a.z, b.x, 1.4, b.z, a.x, 1.4, a.z);
  }
  strokeWall.geometry.setAttribute('position', new THREE.Float32BufferAttribute(v, 3));
}
function clearStroke() { S.stroke = []; renderStroke(); updateStrokeReport(); }

function updateStrokeReport() {
  if (!S.data) return;
  const el = $('player-shield-text');
  const { train, test } = S.data;
  let html = '';
  if (S.stroke.length >= 2) {
    const poly = S.stroke.map((p) => [fromW(p.x), fromW(p.z)]);
    const clf = strokeClassifier(poly, train.X, train.y);
    const tr = clf.acc, te = clf.evaluate(test.X, test.y);
    html += `Trazo libre → train <b>${Math.round(tr * 100)}%</b> · test <b>${Math.round(te * 100)}%</b>. `;
    if (S.world === 2 && tr < 0.8) html += '<br>Una línea no separa los anillos: <b>eleva el terreno</b>.';
    else if (tr > 0.95) html += '<br>¡Buen escudo!';
  } else html += 'Activa «Trazar escudo» y dibuja sobre la meseta. ';
  if (S.world === 2) {
    const f = (X, y) => { let ok = 0; X.forEach((p, i) => { const r2 = (p[0] - Math.PI / 2) ** 2 + (p[1] - Math.PI / 2) ** 2; ok += ((r2 < S.plane ? 1 : -1) === y[i]); }); return ok / X.length; };
    html += `<br>Plano z=${S.plane.toFixed(2)} (elevado) → train <b>${Math.round(f(train.X, train.y) * 100)}%</b> · test <b>${Math.round(f(test.X, test.y) * 100)}%</b>`;
  }
  el.innerHTML = html;
}

// ------------------------------------------------------------------ UI
function toast(msg, cls = '') {
  const t = $('toast'); t.className = `toast show ${cls}`; t.textContent = msg;
  clearTimeout(toast._t); toast._t = setTimeout(() => (t.className = 'toast'), 3600);
}
$('btn-stroke').addEventListener('click', (e) => {
  S.strokeMode = !S.strokeMode; e.currentTarget.classList.toggle('on', S.strokeMode);
  controls.enabled = !S.strokeMode; canvas.classList.toggle('drawing', S.strokeMode);
  if (S.strokeMode) { toast('✍ Dibuja sobre la meseta para trazar tu escudo'); }
});
$('btn-clear').addEventListener('click', clearStroke);
$('btn-wave').addEventListener('click', nisqWave);
$('btn-test').addEventListener('click', () => {
  if (S.cfgIdx < 0) return;
  castSpell(S.cfgIdx); toast('🧪 Test de Overfitting en curso…');
});
$('btn-points').addEventListener('click', (e) => {
  S.showTest = !S.showTest; e.currentTarget.classList.toggle('on', S.showTest); testMeshes.forEach((m) => (m.visible = S.showTest));
});
$('lift').addEventListener('input', (e) => { S.lift = +e.target.value; });
$('plane').addEventListener('input', (e) => { S.plane = +e.target.value; updateStrokeReport(); });
document.querySelectorAll('.world-tab:not(.locked)').forEach((b) => b.addEventListener('click', () => selectWorld(+b.dataset.world)));

// ------------------------------------------------------------------ bucle
function resize() {
  const w = innerWidth, h = innerHeight; renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix();
}
addEventListener('resize', resize); resize();

const clock = new THREE.Clock();
let finalized = false;
function frame() {
  const dt = Math.min(clock.getDelta(), 0.05), t = clock.elapsedTime;
  if (S.data) {
    if (S.morphT < 1) {
      S.morphT = Math.min(1, S.morphT + dt / 1.3);
      const e = S.morphT < 0.5 ? 4 * S.morphT ** 3 : 1 - Math.pow(-2 * S.morphT + 2, 3) / 2;
      for (let k = 0; k < N * N; k++) S.hCur[k] = S.hFrom[k] + (S.hTo[k] - S.hFrom[k]) * e;
      applyTerrain();
    }
    // presión del vórtice: el esfuerzo sube hasta su valor objetivo
    if (shield.ready && S.shatterT < 0 && S.formT > 0.9) {
      S.stressT = Math.min(S.stressTarget, S.stressT + dt * 0.55);
      if (S.stressT >= 1 && S.stressTarget >= 1) { shatter(); if (!S.done) { S.done = true; finalizeSpell(); } }
      else if (!S.done && S.stressT >= S.stressTarget - 1e-6) { S.done = true; finalizeSpell(); }
    }
    if (S.waveT >= 0) { S.waveT += dt; S.nisq = Math.max(0, 1 - S.waveT / 1.2); if (S.waveT > 1.2) S.waveT = -1; }
    const pct = Math.min(1.2, S.stressT) * 100;
    $('stress-fill').style.width = `${Math.min(100, pct)}%`; $('stress-val').textContent = `${Math.round(pct)}%`;
    if (S.shatterT < 0 && S.stressT > 0.6 && S.stressT < 1) setStressText('warn', 'Grietas en el cristal… el kernel se está pegando al ruido.');
    placePoints(t); updateShield(dt, t);
    tMat.emissive.setHex(S.shatterT >= 0 && S.shatterT < 0.4 ? 0x551122 : 0x000000);
  }
  controls.update(); renderer.render(scene, camera); requestAnimationFrame(frame);
}

selectWorld(2).then(() => { $('loading').classList.add('done'); frame(); });
