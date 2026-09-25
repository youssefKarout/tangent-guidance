# Results

All numbers below are **paired**: every branch starts from the *same* `x0` bank
(seed 0), so a difference is attributable to the branch that changed.

Setup: MNIST · frozen VAE `8×7×7` (latent std 0.434) · frozen FM, 50 Euler steps
for generation · retrieval bank = 60 000 *train* latents · FID (Inception-V3
pool3) against 1 000 *test* images · n = 512 samples · CPU.

## A. Which objective should the teacher use?

Same step size (η = 0.35), same m = 3 re-aimed steps, only the objective changes:

| teacher objective | FID | Δ vs FM | latent kNN distance |
|---|---|---|---|
| — (FM only) | 43.208 | — | 18.083 |
| `latent_bary` — the "obvious" one | 46.759 | **+3.551** | 12.410 |
| `pixel_bary` (decoded-space barycenter) | 44.780 | +1.572 | 15.679 |
| `pixel_nn1` (nearest real image, no averaging) | 43.568 | +0.360 | 16.573 |
| **`pixel_tangent`** (radial part removed) | **42.729** | **−0.479** | 17.806 |
| VAE round-trip `decode(encode(real))` | 33.821 | — | ceiling at 23.78 dB PSNR |

The ladder is monotone in one variable: how much of the *mean-seeking* pull is
removed. Note that the best configuration has the **worst** (largest) kNN
distance to the bank — it is not "getting closer to real data" per sample,
it is redistributing toward a better-shaped distribution.

## B. Step size and re-aiming (`pixel_tangent`)

| setting | drift | FID | Δ vs FM | FID mean term | FID trace term |
|---|---|---|---|---|---|
| FM only | 0.000 | 43.208 | — | 11.321 | 31.887 |
| `once`, η=0.35 | 0.350 | 43.111 | −0.098 | 11.286 | 31.825 |
| `once`, η=0.70 | 0.700 | 42.932 | −0.276 | 11.278 | 31.654 |
| `iter`, η=0.15 | 0.4 | 43.056 | −0.152 | — | — |
| `iter`, η=0.35 | 1.033 | 42.729 | −0.479 | 11.199 | 31.531 |
| `iter`, η=0.70 | 1.963 | 41.874 | −1.334 | 10.774 | 31.100 |
| **`iter`, η=1.00** | 2.660 | **41.592** | **−1.616** | 10.526 | 31.066 |
| `iter`, η=1.40 | 3.411 | 41.781 | −1.427 | 10.636 | 31.145 |

Three things worth noticing:

1. **Both** FID terms bottom out at η = 1.0, so it is a genuine optimum.
2. Re-aiming matters more than distance: at the same per-step length (η = 0.7)
   a single nudge gives −0.28, three re-aimed steps give **−1.33**.
3. Nothing collapses: pixel std / sharpness stay at the FM level
   (0.3152 / 0.1544 → 0.3140 / 0.1525 at η = 1.0).

## C. Who does the teacher help? (per-sample, n = 1024)

Per-sample badness = distance to the k nearest real images in Inception feature
space (the FID-aligned metric). Samples are binned by how bad the FM endpoint was.

| branch | worst 10% Δ (improved) | best 10% Δ (improved) | Spearman(badness, Δ) |
|---|---|---|---|
| teacher η=0.35 | −0.081 (59%) | +0.094 (37%) | −0.14 |
| teacher η=0.70 | −0.355 (71%) | +0.283 (29%) | −0.26 |
| teacher η=1.40 | −0.727 (73%) | +0.534 (22%) | −0.34 |
| *trained v1 field* | +0.016 (40%) | +0.134 (33%) | −0.11 |

The teacher is a **tail repairer with a diversity side-effect**: broken samples
are pulled toward real data, while already-good samples are pushed slightly off
their nearest real image (spread), which is exactly the trace-term improvement
seen in B. On average the per-sample distance barely moves (9.397 → 9.435), yet
FID improves by 1.6 — the gain is in the distribution's shape, not its centre.


## D. Why the field cannot learn the tangent direction (probe_learnability)

The distilled parametric correction field (`FieldNet`) loss stayed flat at ~5.0.
`probe_learnability.py` measures the target signal's learnability directly:

| target objective | ‖mean direction‖ | split-half cosine | train 1−cos | val 1−cos | learnable? |
|---|---|---|---|---|---|
| `latent_bary` | 0.173 | **0.970** | 3.34 | 3.37 | **yes** (smooth global drift) |
| `pixel_tangent` | 0.029 | **0.202** | 4.96 | 4.99 | **no** (essentially random) |
| `pixel_tangent_k` | 0.030 | 0.425 | 4.91 | 4.97 | **no** |

Conclusion: the teacher works precisely *because* it looks at the real data bank at inference.
Parametric distillation from the latent alone lacks the necessary retrieval context.

