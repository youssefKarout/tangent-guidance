# Flow-Matching + Retrieval Teacher on MNIST Latents

A two-stage latent generative model where **Stage B is a retrieval step, not a
learned network**:

```
x0 ~ N(0,I)  --[frozen FM, 50 Euler steps]-->  x̂          (Stage A: the generator)
                                              |
                    teacher probe: roll out, decode, look up the
                    nearest real images in the training bank, and step
                    along the local data manifold (no mean-seeking)
                                              |
                     x̂ + η·û  (m times, re-aimed)  -->  x_ref     (Stage B)
```

Everything runs on MNIST in a frozen-VAE latent space (`8×7×7`). Stage A is a
plain flow-matching model; Stage B consults the training set at inference and
improves FID by **−1.62** (paired, 512 samples) over the plain FM output, with
**no additional learned parameters**.

| stage | what it is | trained? |
|---|---|---|
| VAE | `8×7×7` autoencoder, frozen afterwards | yes (`train_vae.py`) |
| Stage A | residual conv velocity field + sinusoidal time embedding | yes (`train_fm.py`) |
| Stage B | **retrieval teacher**: tangent step toward nearest real images | **no** |

## Quickstart

```bash
pip install -r requirements.txt

# 1) VAE (~minutes on CPU)
python train_vae.py --data-dir data --epochs 30 --batch 256 --out-dir vae_run

# 2) Stage A: FM on the frozen latents
python train_fm.py --vae-checkpoint vae_run/vae.pt --data-dir data \
    --epochs 30 --batch 256 --out-dir trained_fm

# 3) inference: FM rollout alone, then with the retrieval teacher (+FID)
python infer.py --vae-checkpoint vae_run/vae.pt \
    --fm-checkpoint trained_fm/fm_ema.pt --data-dir data --offline \
    --n 64 --no-teacher --out-dir inference_fm

python infer.py --vae-checkpoint vae_run/vae.pt \
    --fm-checkpoint trained_fm/fm_ema.pt --data-dir data --offline \
    --n 512 --etas 1.0,0.7 --m 3 --teacher-space pixel_tangent \
    --fid --n-real 1000 --out-dir inference_teacher
```

`infer.py` writes sample grids, a paired comparison strip (identical `x0` in
every row) and `summary.json` with FID (mean/trace split), the VAE round-trip
floor, drift and latent kNN distances.

## Results (paired, 512 samples, FID vs 1000 real test images)

| configuration | FID | Δ vs FM |
|---|---|---|
| FM rollout only (Stage A) | 43.21 | — |
| VAE round-trip `decode(encode(real))` | **33.82** | latent-space ceiling (23.8 dB) |
| + teacher, `latent_bary` (the "obvious" objective) | 46.76 | **+3.55** |
| + teacher, `pixel_bary` | 44.78 | +1.57 |
| + teacher, `pixel_nn1` | 43.57 | +0.36 |
| **+ teacher, `pixel_tangent` (η=1.0, m=3)** | **41.59** | **−1.62** |

Step-size curve for `pixel_tangent` (m=3, direction re-aimed after every step):

| η | 0.35 | 0.70 | 1.00 | 1.40 |
|---|---|---|---|---|
| ΔFID, `once` (single nudge) | −0.10 | −0.28 | — | — |
| ΔFID, `iter` (re-aimed, m=3) | −0.48 | −1.33 | **−1.62** | −1.43 |

Full tables, the per-sample tail analysis and the failed distillation
experiments are in [RESULTS.md](RESULTS.md).

## Why the teacher looks like this

The teacher steps in **decoded-pixel space**, through the frozen decoder, and
removes the component that points at the neighbourhood mean:

```
p̂  = decode(x̂)                        look at the sample as an image
p̄  = mean of the k decoded neighbours      local "average digit"
û  = normalize(p̂ − p̄)                 the outward, mean-seeking direction
d  = y_nearest − p̂                     pull toward the closest real image
d ⊖= (d·û)û                           drop the radial (shrink) part
step  = −∇_x ‖decode(x̂) − (p̂ + d)‖²    back to a latent direction
```

Two changes turn +3.55 into −1.62: measuring in a space the decoder preserves,
and **removing the mean-seeking component**. Pulling toward a neighbourhood
*mean* does minimise the distance to the data cloud, but it makes samples
central and blurry — the per-sample analysis in RESULTS.md shows that trade
explicitly (the worst samples improve, the best get slightly spread out).

## Honest limitation (why Stage B is not a learned field)

We did try to distil the teacher into a small per-timestep correction field. It
cannot work as stated, and `probe_learnability.py` measures why: the tangent
direction is **not a function of the sample alone**.

| objective | ‖mean direction‖ | split-half cosine | can a field learn it? |
|---|---|---|---|
| `latent_bary` (distillable, but FID-hurting) | 0.173 | **0.970** | yes (held-out 1−cos 5.00 → 3.37) |
| `pixel_tangent` (FID-improving) | 0.029 | **0.202** | **no** (flat at 5.00) |
| `pixel_tangent_k` | 0.030 | 0.425 | no (flat) |

A direction whose per-sample variation is dominated by *which* real image is
nearest is only available to a method that can consult the training set at
inference. So this repo ships the retrieval step, and the distilled-field route
is documented as the negative result it is.

## Files

| file | role |
|---|---|
| `models.py` | VAE + FM architectures and checkpoint loaders (single source of truth) |
| `fm_ops.py` | MNIST data, encode/decode, FM integration, kNN helpers |
| `teacher.py` | the retrieval teacher: 5 objectives + `Teacher.corr_at` |
| `infer.py` | **the pipeline**: Stage A + Stage B, previews, FID, `summary.json` |
| `fid.py` | Inception pool3 features, FID and its mean/trace split |
| `probe_learnability.py` | evidence that the tangent direction is not distillable |
| `train_vae.py`, `train_fm.py` | training for the two frozen stages |
| `RESULTS.md` | all measurements, including the failed learned-field attempts |

## Reproducing the numbers

```bash
# step-size curve (and the peak row of the FID table)
python infer.py --offline --device cuda --n 512 --etas 0.35,0.7,1.0,1.4 \
    --m 3 --teacher-space pixel_tangent --fid --n-real 1000

# the objective comparison, one run per space, same paired x0 bank
for s in latent_bary pixel_bary pixel_nn1 pixel_tangent; do
  python infer.py --offline --n 512 --etas 0.35 --teacher-space $s \
      --fid --n-real 1000 --out-dir gate_$s
done

# the distillability probe
python probe_learnability.py --offline --n 1024 --spaces latent_bary,pixel_tangent
```

Numbers depend on the frozen VAE/FM quality: the round-trip floor (FID 33.82 at
23.8 dB) is the lower bound any latent-space stage can reach with this VAE, and
the FM baseline is 43.21. On a CPU-only box expect ~4 min per 512-sample FID run.


