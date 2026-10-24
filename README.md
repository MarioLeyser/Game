# Quantum Defense

Juego educativo de Quantum Machine Learning. MVP: **Mundo 2 — La Meseta Radial** y **Mundo 3 — Las Cumbres del Vórtice**.

- Terreno 3D que se deforma con la superficie de decisión de cada kernel.
- **Trazo del escudo hiperplano** + elevación radial `z = x²+y²` (Mundo 2).
- **Libro de Hechizos** (kernels lineal / polinómico / RBF / ZZFeatureMap+FidelityQuantumKernel).
- **Test de Overfitting**: el escudo de cristal se rompe si sobreajustas al Slime infiltrado → [docs/OVERFITTING.md](docs/OVERFITTING.md).

## Estructura
```
backend/quantum_defense/   Precálculo con Qiskit ML (ZZFeatureMap + FidelityQuantumKernel + QSVC)
web/                       App estática (Three.js) + web/data/*.json precalculados
.github/workflows/pages.yml  Despliegue a GitHub Pages (publica web/)
```

## Regenerar datos
```
python -m venv .venv && .venv\Scripts\activate
pip install -r backend/requirements.txt
cd backend && python -m quantum_defense.precompute --out ../web/data --grid 32 --verify
```

## Ejecutar local
```
cd web && python -m http.server 8000   # http://localhost:8000
```
