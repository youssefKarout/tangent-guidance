"""End-to-end inference: flow-matching rollout plus the retrieval teacher.

    Stage A   x0 ~ N(0, I) --frozen FM, coarse_steps Euler steps--> x
    Stage B    x --teacher.corr_at, m re-aimed steps of size eta--> x_ref

Stage A is the generator. Stage B is optional, learns nothing, and consults the
frozen training bank (see teacher.py). The run writes sample grids, a paired
comparison strip and summary.json with FID, the VAE round-trip floor, drift and
latent kNN distances; --save-latents also stores the raw latents.

    # FM only (Stage A baseline), no teacher and no FID (fast)
    python infer.py --vae-checkpoint vae_run/vae.pt \\
        --fm-checkpoint trained_fm/fm_ema.pt --data-dir data --offline \\
        --n 64 --no-teacher --out-dir inference_fm

    # FM vs teacher at eta = 1.0 and 0.7, with FID
    python infer.py --vae-checkpoint vae_run/vae.pt \\
        --fm-checkpoint trained_fm/fm_ema.pt --data-dir data --offline \\
        --n 64 --etas 1.0,0.7 --teacher-space pixel_tangent --m 3 \\
        --fid --n-real 1000 --out-dir inference_out
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import json
import time
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fm_ops import (decode_data, encode_data, fm_integrate, knn_bary_dist,
                    load_mnist, log, t_at)
from models import load_fm, load_vae
from teacher import Teacher, build_pixel_bank
from fid import InceptionFeats, fid_parts


# Stage A: noise -> latent with the frozen FM
@torch.no_grad()
def stage_a(fm, x0, coarse_steps, device, batch):
    outs = []
    for i in range(0, x0.shape[0], batch):
        xb = x0[i:i + batch].to(device)
        outs.append(fm_integrate(xb, t_at(xb.shape[0], 0.0, device), fm,
                                 steps=coarse_steps).cpu())
    return torch.cat(outs, dim=0)


# Stage B: retrieval correction, m re-aimed steps
@torch.no_grad()
def stage_b(teacher, x_start, eta, m, t_ref, device, batch, diag=True):
    """Returns (final latents, per-step diagnostics)."""
    outs, rows = [], []
    for i in range(0, x_start.shape[0], batch):
        x = x_start[i:i + batch].to(device)
        x0b = x.clone()
        for step in range(m):
            r = teacher.corr_at(x, t_at(x.shape[0], t_ref, device))
            if diag:
                rows.append({
                    "step": step + 1,
                    "drift": float((x - x0b).flatten(1).norm(dim=-1).mean()),
                    "grad_norm": float(r["grad"].flatten(1).norm(dim=-1).mean()),
                    "base": float(r["base"].mean()),
                    **{k: float(v) for k, v in r["diag"].items()},
                })
            x = x + eta * r["corr"]
        outs.append(x.cpu())
    return torch.cat(outs, dim=0), rows


def save_grid(latents, vae, path, title, ncol=8, n=64):
    n = min(n, latents.shape[0])
    nrow = int(np.ceil(n / ncol))
    imgs = ((decode_data(vae, latents[:n]) + 1.0) / 2.0).clamp(0, 1)
    fig, axes = plt.subplots(nrow, ncol, figsize=(1.1 * ncol, 1.2 * nrow))
    for j, ax in enumerate(np.atleast_1d(axes).ravel()):
        ax.axis("off")
        if j < n:
            ax.imshow(imgs[j, 0], cmap="gray", vmin=0, vmax=1)
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


# FID for a dict of {name: latents} against the real MNIST test images
def score_fid(branches, vae, test_x, n_real, inception_batch, device,
              roundtrip=False, split=True):
    inc = InceptionFeats(device)
    real = test_x[:n_real].reshape(-1, 1, 28, 28)     # [0,1], canonical shape
    log(f"[inception] real reference: {tuple(real.shape)}")
    f_real = inc.feats(real, batch=inception_batch)
    out = {}
    for name, lat in branches.items():
        img = ((decode_data(vae, lat) + 1.0) / 2.0).clamp(0, 1)
        f = inc.feats(img, batch=inception_batch)
        if split:
            total, mean_t, trace_t = fid_parts(f, f_real)
            out[name] = {"fid": total, "fid_mean_term": mean_t,
                         "fid_trace_term": trace_t, "n": int(lat.shape[0])}
        else:
            from fid import fid_from_feats
            out[name] = {"fid": fid_from_feats(f, f_real), "n": int(lat.shape[0])}
        log(f"[FID] {name:<22} = {out[name]['fid']:7.3f}"
            + (f"  (mean {mean_t:6.2f} + trace {trace_t:6.2f})" if split else ""))
    if roundtrip:
        rt = encode_data(vae, real * 2 - 1, device)
        rec = decode_data(vae, rt)
        p = ((rec + 1.0) / 2.0).clamp(0, 1)
        mse = float(((p - real) ** 2).mean())
        psnr = float(10 * np.log10(1.0 / max(mse, 1e-12)))
        f = inc.feats(p, batch=inception_batch)
        total, mean_t, trace_t = fid_parts(f, f_real)
        out["vae_roundtrip"] = {"fid": total, "psnr_db": psnr, "mse01": mse,
                                "fid_mean_term": mean_t,
                                "fid_trace_term": trace_t}
        log(f"[floor] VAE round-trip decode(encode(real)): FID {total:.3f} at "
            f"{psnr:.2f} dB PSNR")
    return out


# CLI
def build_parser():
    p = argparse.ArgumentParser(
        description="Inference: FM rollout + retrieval teacher")
    p.add_argument("--vae-checkpoint", type=str, default=os.path.join("vae_run", "vae.pt"))
    p.add_argument("--fm-checkpoint", type=str, default=os.path.join("trained_fm", "fm_ema.pt"))
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--out-dir", type=str, default="inference_out")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--use-synthetic", action="store_true",
                   help="random blobs instead of MNIST (code-path check only)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n", type=int, default=64, help="samples to generate")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--coarse-steps", type=int, default=50,
                   help="Stage-A Euler steps from t=0 to 1")
    p.add_argument("--no-teacher", action="store_true", help="Stage A only")
    # Retrieval teacher (Stage B)
    p.add_argument("--teacher-space", type=str, default="pixel_tangent",
                   help="latent_bary | pixel_bary | pixel_nn1 | pixel_tangent | pixel_tangent_k")
    p.add_argument("--etas", type=str, default="1.0",
                   help="comma list of stage-B step sizes to evaluate")
    p.add_argument("--m", type=int, default=3, help="teacher steps per eta")
    p.add_argument("--teacher-apply", type=str, default="iter",
                   choices=["iter", "once"],
                   help="iter = re-aim the teacher after every step (default), "
                        "once = a single nudge of size eta")
    p.add_argument("--t-ref", type=float, default=0.95,
                   help="time at which the teacher is applied")
    p.add_argument("--teacher-steps", type=int, default=8,
                   help="FM rollout steps inside the teacher probe")
    p.add_argument("--knn-k", type=int, default=25, help="neighbours in the bank")
    p.add_argument("--prefilter", type=int, default=256,
                   help="latent candidates before the pixel-space kNN (0 = exact)")
    p.add_argument("--bank-split", type=str, default="train", choices=["train", "test"])
    p.add_argument("--n-bank", type=int, default=0, help="cap the bank (0 = all)")
    p.add_argument("--cache-dir", type=str, default="cache")
    # FID
    p.add_argument("--fid", action="store_true", help="also compute FID vs the real test set")
    p.add_argument("--n-real", type=int, default=1000)
    p.add_argument("--inception-batch", type=int, default=128)
    p.add_argument("--no-roundtrip-floor", action="store_true",
                   help="skip the VAE round-trip FID floor")
    p.add_argument("--save-latents", action="store_true")
    return p



def main():
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    t0 = time.time()

    # ---- data (train = the retrieval bank, test = the FID reference)
    if args.use_synthetic:
        from fm_ops import make_synthetic_mnist
        train_x, _ = make_synthetic_mnist(8000, seed=0)
        test_x, _ = make_synthetic_mnist(4000, seed=1)
        train_x = train_x.reshape(-1, 1, 28, 28)
        test_x = test_x.reshape(-1, 1, 28, 28)
        log("[data] SYNTHETIC blobs -> code-path check only")
    else:
        train_x, _, test_x, _ = load_mnist(args.data_dir, args.offline)
    ref = train_x if args.bank_split == "train" else test_x

    # Frozen models
    vae, shape = load_vae(args.vae_checkpoint, device)
    fm = load_fm(args.fm_checkpoint, device, shape)
    log(f"[frozen] VAE -> latent {shape} | FM {args.fm_checkpoint}")

    # Retrieval bank: training latents (decoded pixels are added if needed)
    cache = os.path.join(args.cache_dir,
                         f"bank_latents_{args.bank_split}_{len(ref)}.pt")
    if os.path.exists(cache):
        bank = torch.load(cache, map_location="cpu", weights_only=False)
        log(f"[bank] latents from cache {tuple(bank.shape)}")
    else:
        bank = encode_data(vae, ref * 2 - 1, device)
        torch.save(bank, cache)
        log(f"[bank] latents encoded + cached {tuple(bank.shape)}")
    if args.n_bank > 0:
        bank = bank[:args.n_bank]
    log(f"[bank] {tuple(bank.shape)} | split={args.bank_split} | "
        f"latent std {bank.std():.4f}")

    # Stage A: one shared x0 bank, then the FM endpoint
    x0 = torch.randn(args.n, *shape)
    fm_lat = stage_a(fm, x0, args.coarse_steps, device, args.batch)
    log(f"[stageA] {args.n} FM endpoints in {time.time() - t0:.0f}s")

    branches = {"fm": fm_lat}
    diag = {}
    if not args.no_teacher:
        pix = None
        if args.teacher_space.startswith("pixel"):
            pix = build_pixel_bank(vae, bank, cache_path=os.path.join(
                args.cache_dir,
                f"bank_pixels_{args.bank_split}_{len(bank)}.pt"))
        teacher = Teacher(fm, bank, space=args.teacher_space, vae=vae,
                          bank_pixels=pix, steps=args.teacher_steps,
                          k=args.knn_k, prefilter=args.prefilter)
        log(f"[teacher] {teacher.meta()} | apply={args.teacher_apply} "
            f"| m={args.m} | t_ref={args.t_ref} | etas={args.etas}")
        for eta_s in [e for e in args.etas.split(",") if e != ""]:
            eta = float(eta_s)
            m = args.m if args.teacher_apply == "iter" else 1
            lat, rows = stage_b(teacher, fm_lat, eta, m, args.t_ref, device,
                                args.batch)
            name = f"teacher_eta{eta:g}"
            branches[name] = lat
            diag[name] = {"eta": eta, "m": m, "rows": rows[-1]}
            d = rows[-1]
            log(f"[stageB] {name}: m={m} drift {d['drift']:.3f} | "
                f"|grad| {d['grad_norm']:.4f} | teacher loss {d['base']:.3f}"
                + (f" | tangent_keep {d['tangent_keep']:.3f}"
                   if "tangent_keep" in d else ""))
    # Preview grids; every branch starts from the same x0, so rows are paired.
    save_grid(fm_lat, vae, os.path.join(args.out_dir, "samples_fm.png"),
              f"Stage A only (FM rollout, {args.coarse_steps} steps)", n=args.n)
    for name, lat in branches.items():
        if name == "fm":
            continue
        save_grid(lat, vae, os.path.join(args.out_dir, f"samples_{name}.png"),
                  f"{name} (space={args.teacher_space})", n=args.n)
    if len(branches) > 1:
        keys = list(branches)[:6]
        k = min(8, args.n)
        fig, axes = plt.subplots(len(keys), k, figsize=(1.0 * k, 1.15 * len(keys)))
        for r, key in enumerate(keys):
            imgs = ((decode_data(vae, branches[key][:k]) + 1.0) / 2.0).clamp(0, 1)
            for c in range(k):
                axes[r, c].axis("off")
                axes[r, c].imshow(imgs[c, 0], cmap="gray", vmin=0, vmax=1)
            axes[r, 0].set_ylabel(key, fontsize=7)
        fig.suptitle("same x0 in every row (paired comparison)", fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "compare_strip.png"), dpi=130)
        plt.close()

    # Optional FID, plus the VAE round-trip floor
    fid = {}
    if args.fid:
        fid = score_fid(branches, vae, test_x, args.n_real, args.inception_batch,
                        device, roundtrip=not args.no_roundtrip_floor)

    # Quantitative summary
    summary = {"args": vars(args),
               "teacher": (None if args.no_teacher else teacher.meta()),
               "bank": {"split": args.bank_split, "n": int(bank.shape[0]),
                        "std": float(bank.std())},
               "knn_bary_dist": {}, "fid": fid, "stageB": diag,
               "elapsed_s": None}
    for name, lat in branches.items():
        summary["knn_bary_dist"][name] = float(
            knn_bary_dist(lat, bank, args.knn_k).mean())
        log(f"[latent] {name:<18} kNN-bary dist "
            f"{summary['knn_bary_dist'][name]:8.3f}")
    fm_fid = fid.get("fm", {}).get("fid")
    if fm_fid is not None:
        summary["fid_delta_vs_fm"] = {}
        for name in branches:
            if name == "fm":
                continue
            d = fid[name]["fid"] - fm_fid
            summary["fid_delta_vs_fm"][name] = d
            log(f"[delta] {name:<18} FID {d:+.3f} vs FM")
    summary["elapsed_s"] = time.time() - t0
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    if args.save_latents:
        torch.save({"x0": x0, "latents": branches, "seed": args.seed},
                   os.path.join(args.out_dir, "samples.pt"))
    log(f"DONE in {summary['elapsed_s']:.0f}s -> {args.out_dir}/summary.json")


if __name__ == "__main__":
    main()



