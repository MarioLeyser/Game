// Matemática del juego: contornos, clasificadores y modelo de esfuerzo del Test de Overfitting.
export const lerp = (a, b, t) => a + (b - a) * t;

/** Interpolación bilineal en un arreglo plano n×n; (u,v) en [0,1], u→columna (x), v→fila (z). */
export function bilinear(arr, n, u, v) {
  const x = Math.min(Math.max(u, 0), 1) * (n - 1), y = Math.min(Math.max(v, 0), 1) * (n - 1);
  const j = Math.min(Math.floor(x), n - 2), i = Math.min(Math.floor(y), n - 2), fx = x - j, fy = y - i;
  return lerp(lerp(arr[i * n + j], arr[i * n + j + 1], fx), lerp(arr[(i + 1) * n + j], arr[(i + 1) * n + j + 1], fx), fy);
}

/** Marching squares sobre dec[i][j] (i→y, j→x). Devuelve segmentos [ax, az, bx, bz] en coordenadas de mundo. */
export function contourSegments(dec, G, min, max, toW) {
  const gx = (k) => min + (k / (G - 1)) * (max - min);
  const out = [];
  for (let i = 0; i < G - 1; i++) for (let j = 0; j < G - 1; j++) {
    const v00 = dec[i][j], v10 = dec[i][j + 1], v01 = dec[i + 1][j], v11 = dec[i + 1][j + 1];
    const pts = [];
    const edge = (va, vb, xa, ya, xb, yb) => {
      if ((va > 0) !== (vb > 0)) { const t = va / (va - vb); pts.push([xa + (xb - xa) * t, ya + (yb - ya) * t]); }
    };
    const x0 = gx(j), x1 = gx(j + 1), y0 = gx(i), y1 = gx(i + 1);
    edge(v00, v10, x0, y0, x1, y0); edge(v10, v11, x1, y0, x1, y1);
    edge(v01, v11, x0, y1, x1, y1); edge(v00, v01, x0, y0, x0, y1);
    for (let k = 0; k + 1 < pts.length; k += 2) out.push([toW(pts[k][0]), toW(pts[k][1]), toW(pts[k + 1][0]), toW(pts[k + 1][1])]);
  }
  return out;
}

/** Valor de decisión en (x,y) de datos con bilinear sobre la malla precalculada. */
export function decisionAt(dec, G, min, max, x, y) {
  const u = ((x - min) / (max - min)) * (G - 1), v = ((y - min) / (max - min)) * (G - 1);
  const j = Math.max(0, Math.min(G - 2, Math.floor(u))), i = Math.max(0, Math.min(G - 2, Math.floor(v)));
  const fx = Math.min(1, Math.max(0, u - j)), fy = Math.min(1, Math.max(0, v - i));
  return lerp(lerp(dec[i][j], dec[i][j + 1], fx), lerp(dec[i + 1][j], dec[i + 1][j + 1], fx), fy);
}

export function accuracyOnGrid(dec, G, min, max, X, y) {
  let ok = 0;
  X.forEach((p, k) => { ok += (decisionAt(dec, G, min, max, p[0], p[1]) > 0 ? 1 : -1) === y[k]; });
  return ok / X.length;
}

/** Clasificador por trazo: el signo lo da el lado del segmento más cercano; orientación óptima sobre train. */
export function strokeClassifier(poly, Xtr, ytr) {
  const side = (p) => {
    let best = Infinity, s = 0;
    for (let i = 0; i < poly.length - 1; i++) {
      const [ax, ay] = poly[i], [bx, by] = poly[i + 1];
      const dx = bx - ax, dy = by - ay, L = dx * dx + dy * dy || 1e-9;
      const t = Math.max(0, Math.min(1, ((p[0] - ax) * dx + (p[1] - ay) * dy) / L));
      const d = Math.hypot(p[0] - (ax + t * dx), p[1] - (ay + t * dy));
      if (d < best) { best = d; s = dx * (p[1] - ay) - dy * (p[0] - ax) > 0 ? 1 : -1; }
    }
    return s;
  };
  const raw = (X, y, flip) => X.reduce((a, p, k) => a + (side(p) * flip === y[k]), 0) / X.length;
  const flip = raw(Xtr, ytr, 1) >= raw(Xtr, ytr, -1) ? 1 : -1;
  return { acc: raw(Xtr, ytr, flip), evaluate: (X, y) => raw(X, y, flip) };
}

/**
 * Esfuerzo del cristal: σ = w₀·gap/G* + w₁·drop/R* + w₂·captured. σ ≥ 1 ⇒ el escudo se quiebra.
 * Debe coincidir con quantum_defense/precompute.py::evaluate_config (ver docs/OVERFITTING.md).
 */
export function liveStress(cfg, rules, drop) {
  const w = rules.stress_w;
  return Math.max(0, w[0] * cfg.gap / rules.gap_limit + w[1] * drop / rules.robust_drop_limit + w[2] * cfg.captured);
}
