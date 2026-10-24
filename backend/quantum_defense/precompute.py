"""Precálculo de fronteras y matrices de kernel de los 4 mundos de Quantum Defense.

Uso:
    python -m quantum_defense.precompute --out ../web/data [--grid 20]

Genera JSON estáticos para que el MVP web corra en tiempo real sin backend:
  - Mundos 1-2: SVM clásico (lineal, polinómico, RBF).
  - Mundos 3-4: QSVC con ZZFeatureMap + FidelityQuantumKernel (reps 1..3).
  - Test de Overfitting: acc. de train/test, robustez ante perturbación NISQ,
    "rotura de escudo" y estrellas (1: train, 2: generalización, 3: parsimonia).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel, linear_kernel
from sklearn.svm import SVC

from .worlds import WORLD_META, build_world

JITTER_SIGMA = 0.12      # radianes: perturbación "onda NISQ"
JITTER_DRAWS = 3
GAP_LIMIT = 0.15         # brecha train-test que rompe el escudo
ROBUST_DROP_LIMIT = 0.12
STRESS_W = (0.4, 0.2, 0.5)  # pesos: brecha, fragilidad NISQ, captura del infiltrado


# --------------------------------------------------------------------------
# Kernels
# --------------------------------------------------------------------------
class KernelFn:
    """Interfaz común: callable (A, B) -> matriz de Gram, con memo para el cuántico."""

    def __init__(self, name, fn, cache=False):
        self.name, self.fn, self.cache, self._memo = name, fn, cache, {}

    def __call__(self, A, B=None):
        B = A if B is None else B
        if not self.cache:
            return self.fn(A, B)
        key = (A.tobytes(), B.tobytes())
        if key not in self._memo:
            self._memo[key] = self.fn(A, B)
        return self._memo[key]


def quantum_kernel(reps: int, entanglement: str = "linear") -> KernelFn:
    from qiskit.circuit.library import ZZFeatureMap
    from qiskit_machine_learning.kernels import FidelityQuantumKernel

    fm = ZZFeatureMap(feature_dimension=2, reps=reps, entanglement=entanglement)
    qk = FidelityQuantumKernel(feature_map=fm)
    return KernelFn(f"zz-reps{reps}", lambda A, B: qk.evaluate(x_vec=A, y_vec=B), cache=True)


def classical_kernel(kind: str, **p) -> KernelFn:
    if kind == "linear":
        return KernelFn("linear", lambda A, B: linear_kernel(A, B))
    if kind == "poly":
        return KernelFn("poly", lambda A, B: polynomial_kernel(A, B, degree=p["degree"], gamma=1.0, coef0=1.0))
    if kind == "rbf":
        return KernelFn("rbf", lambda A, B: rbf_kernel(A, B, gamma=p["gamma"]))
    raise ValueError(kind)


# --------------------------------------------------------------------------
# Configuraciones por mundo (el jugador elige "báculo" y "regularización")
# --------------------------------------------------------------------------
def configs_for(world: int):
    if world == 1:
        return [dict(id=f"linear-C{C}", label=f"SVM lineal C={C}", C=C, complexity=1,
                     kernel=("linear", {})) for C in (0.1, 1.0, 100.0)]
    if world == 2:
        out = [dict(id="poly-d2", label="Polinómico grado 2", C=1.0, complexity=2,
                    kernel=("poly", {"degree": 2})),
               dict(id="poly-d6", label="Polinómico grado 6", C=100.0, complexity=6,
                    kernel=("poly", {"degree": 6}))]
        for g, c in ((0.5, 1.0), (2.0, 1.0), (30.0, 1000.0)):
            out.append(dict(id=f"rbf-g{g}", label=f"RBF gamma={g}", C=c,
                            complexity=1 if g <= 2 else 10, kernel=("rbf", {"gamma": g})))
        return out
    out = []
    for reps in (1, 2, 3):
        for C in (1.0, 1000.0):
            out.append(dict(id=f"zz-r{reps}-C{int(C)}", label=f"ZZFeatureMap reps={reps}, C={int(C)}",
                            C=C, complexity=reps, reps=reps, kernel=("zz", {"reps": reps})))
    return out


# --------------------------------------------------------------------------
# Test de Overfitting físico
# --------------------------------------------------------------------------
def boundary_roughness(dec: np.ndarray) -> float:
    """Cruces de signo por celda: proxy de 'arrugas' del escudo."""
    s = np.sign(dec)
    return float((np.abs(np.diff(s, axis=0)).sum() + np.abs(np.diff(s, axis=1)).sum()) / 2) / dec.shape[0]


def evaluate_config(cfg, kfn, Xtr, ytr, Xte, yte, Xgrid, G, rng, outl):
    Ktr = kfn(Xtr)
    svm = SVC(kernel="precomputed", C=cfg["C"]).fit(Ktr, ytr)  # ≡ QSVC con kernel precomputado
    train_acc = float((svm.predict(Ktr) == ytr).mean())
    test_acc = float((svm.predict(kfn(Xte, Xtr)) == yte).mean())

    # Onda NISQ: perturba los Slimes de prueba y mide cuánto resiste el escudo.
    accs = []
    for _ in range(JITTER_DRAWS):
        Xj = np.clip(Xte + rng.normal(0, JITTER_SIGMA, Xte.shape), 0, np.pi)
        accs.append((svm.predict(kfn(Xj, Xtr)) == yte).mean())
    robust_acc = float(np.mean(accs))

    dec = svm.decision_function(kfn(Xgrid, Xtr)).reshape(G, G)
    rough = boundary_roughness(dec)
    norm_w = float(np.sqrt(max(
        (svm.dual_coef_ @ Ktr[np.ix_(svm.support_, svm.support_)] @ svm.dual_coef_.T).item(), 1e-12)))
    margin = 2.0 / norm_w

    gap = train_acc - test_acc
    drop = test_acc - robust_acc
    # Memorización del Slime infiltrado: el modelo acata la etiqueta corrupta.
    captured = float((svm.predict(Ktr)[outl] == ytr[outl]).mean()) if outl.any() else 0.0
    # Esfuerzo del cristal (>=1 => se quiebra). Ver docs/OVERFITTING.md
    stress = (STRESS_W[0] * gap / GAP_LIMIT + STRESS_W[1] * drop / ROBUST_DROP_LIMIT
              + STRESS_W[2] * captured)
    shattered = bool(stress >= 1.0)
    s1 = train_acc >= 0.85
    s2 = s1 and test_acc >= 0.85 and not shattered
    s3 = s2 and cfg["complexity"] <= 2
    return {
        "id": cfg["id"], "label": cfg["label"], "kernel": cfg["kernel"][0], "C": cfg["C"],
        "params": cfg["kernel"][1], "complexity": cfg["complexity"],
        "train_acc": round(train_acc, 3), "test_acc": round(test_acc, 3),
        "robust_acc": round(robust_acc, 3), "gap": round(gap, 3), "margin": round(margin, 3),
        "roughness": round(rough, 3), "n_support": int(len(svm.support_)),
        "captured": round(captured, 3), "stress": round(float(stress), 3),
        "support_idx": svm.support_.tolist(), "shattered": shattered,
        "stars": int(s1) + int(s2) + int(s3),
        "decision": np.round(dec, 3).tolist(),
        "kernel_train": np.round(Ktr, 4).tolist(),
    }


def run_world(world: int, G: int, seed: int):
    Xtr, ytr, Xte, yte, outl = build_world(world, seed=seed)
    gx = np.linspace(-0.2, np.pi + 0.2, G)
    xx, yy = np.meshgrid(gx, gx)  # dec[i, j] -> (x=gx[j], y=gx[i])
    Xgrid = np.c_[xx.ravel(), yy.ravel()]
    rng = np.random.default_rng(seed)
    kernels: dict = {}
    results = []
    for cfg in configs_for(world):
        kind, p = cfg["kernel"]
        key = (kind, tuple(sorted(p.items())))
        if key not in kernels:
            t = time.time()
            kernels[key] = quantum_kernel(p["reps"]) if kind == "zz" else classical_kernel(kind, **p)
            tag = "cuántico" if kind == "zz" else "clásico"
            print(f"  [{tag}] {cfg['label']}", end="", flush=True)
            results.append(evaluate_config(cfg, kernels[key], Xtr, ytr, Xte, yte, Xgrid, G, rng, outl))
            print(f"  ({time.time() - t:.1f}s) stars={results[-1]['stars']} stress={results[-1]['stress']}")
        else:
            results.append(evaluate_config(cfg, kernels[key], Xtr, ytr, Xte, yte, Xgrid, G, rng, outl))
            print(f"  [cache]    {cfg['label']} stars={results[-1]['stars']}")
    best = max(results, key=lambda r: (r["stars"], r["robust_acc"], -r["complexity"]))
    meta = WORLD_META[world]
    return {
        "world": world, **meta,
        "grid": {"size": G, "min": -0.2, "max": float(np.pi + 0.2)},
        "train": {"X": np.round(Xtr, 4).tolist(), "y": ytr.tolist(), "outlier": outl.tolist()},
        "test": {"X": np.round(Xte, 4).tolist(), "y": yte.tolist()},
        "configs": results, "recommended": best["id"],
        "overfit_rules": {"gap_limit": GAP_LIMIT, "robust_drop_limit": ROBUST_DROP_LIMIT,
                          "jitter_sigma": JITTER_SIGMA, "stress_w": list(STRESS_W)},
    }


def verify_qsvc(world_json: dict):
    """Comprueba que QSVC (API oficial) coincide con el SVC precalculado."""
    from qiskit.circuit.library import ZZFeatureMap
    from qiskit_machine_learning.algorithms import QSVC
    from qiskit_machine_learning.kernels import FidelityQuantumKernel

    cfg = next(c for c in world_json["configs"] if c["kernel"] == "zz" and c["id"] == world_json["recommended"])
    fm = ZZFeatureMap(feature_dimension=2, reps=cfg["params"]["reps"], entanglement="linear")
    qsvc = QSVC(quantum_kernel=FidelityQuantumKernel(feature_map=fm), C=cfg["C"])
    Xtr, ytr = np.array(world_json["train"]["X"]), np.array(world_json["train"]["y"])
    Xte, yte = np.array(world_json["test"]["X"]), np.array(world_json["test"]["y"])
    qsvc.fit(Xtr, ytr)
    acc = float((qsvc.predict(Xte) == yte).mean())
    print(f"  QSVC oficial: test_acc={acc:.3f} (precalculado {cfg['test_acc']})")
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../web/data")
    ap.add_argument("--grid", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--worlds", type=int, nargs="*", default=[1, 2, 3, 4])
    ap.add_argument("--verify", action="store_true", help="valida contra QSVC oficial")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    index = []
    for w in a.worlds:
        print(f"Mundo {w}: {WORLD_META[w]['name']}")
        data = run_world(w, a.grid, a.seed)
        if a.verify and w in (3, 4):
            verify_qsvc(data)
        (out / f"world{w}.json").write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
        index.append({"world": w, "id": data["id"], "name": data["name"], "file": f"world{w}.json",
                      "recommended": data["recommended"]})
    (out / "index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Listo -> {out.resolve()}")


if __name__ == "__main__":
    main()
