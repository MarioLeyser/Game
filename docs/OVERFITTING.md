# Test de Overfitting — matemática y física del cristal

El escudo hiperplano es un **cristal**. Cuanto más se pega el kernel al ruido (el *Slime infiltrado*,
un punto de entrenamiento con etiqueta invertida), más tensión acumula y, al superar su límite,
se rompe en añicos.

## 1. Magnitudes medidas (por hechizo = kernel + C)

| Símbolo | Definición | Significado de juego |
|---|---|---|
| `acc_tr`, `acc_te` | exactitud en train / test | lo que el escudo "cree" vs. realidad |
| `gap = acc_tr − acc_te` | brecha de generalización | tensión estática del cristal |
| `drop = acc_te − acc_rob` | caída ante la **Onda NISQ** (ruido gaussiano σ=0.12 rad sobre los Slimes de prueba, media de 3 sorteos) | fragilidad ante vibración |
| `captured ∈ [0,1]` | fracción de infiltrados que el modelo clasifica con su etiqueta *corrupta* | memorización directa del ruido |
| `margin = 2/‖w‖_H` | margen en el espacio de Hilbert, `‖w‖² = αᵀ K_sv α` | grosor del cristal |
| `roughness` | cruces de signo de la decisión por celda | "arrugas" visibles del escudo |

## 2. Esfuerzo del cristal

```
σ = w₀·(gap / G*) + w₁·(drop / R*) + w₂·captured
G* = 0.15,  R* = 0.12,  (w₀, w₁, w₂) = (0.4, 0.2, 0.5)
```

* `σ < 1` → el cristal resiste (tensión visible: color cian → ámbar → rojo y vibración creciente).
* `σ ≥ 1` → **fractura**.
* Estrellas: ★ train ≥ 85 %, ★★ además test ≥ 85 % y sin fractura, ★★★ además complejidad ≤ 2 (parsimonia).

Con `captured = 1` el término de memorización aporta 0.5, de modo que solo falta
`0.4·gap/G* + 0.2·drop/R* ≥ 0.5` (p. ej. brecha ≈ 0.19 sola, o una combinación brecha + fragilidad NISQ).
Un hechizo que ignora al infiltrado (`captured = 0`) necesita `gap` y `drop` extremos para romperse.

Los mismos coeficientes viven en [`backend/quantum_defense/precompute.py`](../backend/quantum_defense/precompute.py)
(`STRESS_W`, `GAP_LIMIT`, `ROBUST_DROP_LIMIT`) y se exportan en `overfit_rules` de cada JSON;
`web/js/physics.js::liveStress` los usa para recalcular el esfuerzo con cada **Onda NISQ** en vivo.

## 3. Física de la rotura (cliente, `web/js/main.js`)

1. El contorno `f(x)=0` de la decisión se extrae con *marching squares* y se extruye en un muro
   de triángulos (cada segmento → 2 *shards*).
2. Mientras `σ < 1`: cada shard vibra con amplitud `A = 0.05·max(0, σ − 0.35)` (+ la Onda NISQ).
3. Al cruzar `σ = 1`: velocidad inicial radial desde el Slime infiltrado,
   `v = f·d̂ + (0, 2..7, 0)`, `f ∈ [1.5, 4.7]`; gravedad `g = 14`, spin aleatorio, y rebote
   inelástico sobre el terreno (`e = 0.3`, fricción 0.7). Retardo aleatorio ≤ 0.12 s para que la
   fractura "se propague". Camera-shake + ruido blanco filtrado (paso alto 2.5 kHz).

## 4. Por qué ocurre con kernels cuánticos

Con `ZZFeatureMap(reps=r)` el mapa `φ(x)` es más expresivo al subir `r`; con `C` grande el SVM
tiene libertad para trazar un hiperplano en ℋ que aísla al infiltrado: la frontera se "arruga" localmente
alrededor del punto corrupto y deja de generalizar. `reps` bajo y `C` moderado actúan como regularización.
