"""VAE trainer for the latent flow-matching pipeline (MNIST).

    pixel x (1x28x28, [-1,1])  --encoder-->  z (C x 7 x 7)
    z                          --decoder-->  x_hat (1x28x28, [-1,1], tanh)

Both loss terms are kept in the same units, nats summed per sample:

    rec_nats = sum over the 784 pixels of -log p(x|z)
    kl_nats  = sum over the C*7*7 latents of KL(q(z|x) || N(0,1))
    loss     = rec_nats + beta * kl_nats

so beta = 1 is the standard VAE objective. beta is ramped from 0 over
--kl-warmup-steps, so the decoder learns to reconstruct before the prior starts
shaping the latent. Reconstruction uses BCE on the probability implied by the
tanh output; MSE is also available.

The weights saved to vae.pt are those with the best test reconstruction MSE
rather than the last epoch: the KL term keeps trading reconstruction for a
smaller KL after the decoder has converged.

Usage:
    python train_vae.py --sanity                        # CPU smoke test
    python train_vae.py --data-dir data --epochs 30 --batch 256
    python train_vae.py --device cuda --epochs 50 --batch 512 --base 64

Outputs in --out-dir (default "vae_run"): vae.pt, vae_meta.json, recon_grid.png,
samples_grid.png, loss_curve.png, latent_stats.png.
"""

import os
import json
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MNIST_MIRRORS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "http://yann.lecun.com/exdb/mnist/",
]
MNIST_FILES = [
    ("train-images-idx3-ubyte.gz", "train_x.npy"),
    ("train-labels-idx1-ubyte.gz", "train_y.npy"),
    ("t10k-images-idx3-ubyte.gz", "test_x.npy"),
    ("t10k-labels-idx1-ubyte.gz", "test_y.npy"),
]


def log(*a):
    print(*a, flush=True)


def _download(url, dest):
    import urllib.request
    log(f"    downloading {url} -> {dest}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as f:
        f.write(resp.read())


def _parse_idx(path, offset):
    import gzip
    import struct
    with gzip.open(path, "rb") as f:
        data = f.read()
    if len(data) == 0:
        raise RuntimeError(f"empty file: {path}")
    magic = struct.unpack(">I", data[:4])[0]
    if magic == 2051:
        n = struct.unpack(">I", data[4:8])[0]
        h = struct.unpack(">I", data[8:12])[0]
        w = struct.unpack(">I", data[12:16])[0]
        arr = np.frombuffer(data, dtype=np.uint8, offset=offset).reshape(n, h * w)
    elif magic == 2049:
        n = struct.unpack(">I", data[4:8])[0]
        arr = np.frombuffer(data, dtype=np.uint8, offset=offset).reshape(n)
    else:
        raise RuntimeError(f"bad idx magic {magic} in {path}")
    return arr


def load_mnist(data_dir):
    """Return (train_x, train_y, test_x, test_y); images float32 in [0,1], flat (N,784)."""
    raw_dir = os.path.join(data_dir, "mnist", "raw")
    npy_dir = os.path.join(data_dir, "mnist")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(npy_dir, exist_ok=True)

    paths = [os.path.join(npy_dir, out) for _, out in MNIST_FILES]
    if all(os.path.exists(p) for p in paths):
        log(f"[data] loading cached MNIST from {npy_dir}")
        return (torch.from_numpy(np.load(paths[0])).float(),
                torch.from_numpy(np.load(paths[1])).long(),
                torch.from_numpy(np.load(paths[2])).float(),
                torch.from_numpy(np.load(paths[3])).long())

    for fname, _ in MNIST_FILES:
        gz = os.path.join(raw_dir, fname)
        if os.path.exists(gz):
            continue
        last = None
        for base in MNIST_MIRRORS:
            try:
                _download(base + fname, gz)
                if os.path.getsize(gz) == 0:
                    os.remove(gz)
                    raise RuntimeError("downloaded 0 bytes")
                break
            except Exception as e:  # noqa: BLE001
                last = e
        else:
            raise RuntimeError(f"failed to download {fname}: {last}")

    train_x = _parse_idx(os.path.join(raw_dir, MNIST_FILES[0][0]), 16) / 255.0
    train_y = _parse_idx(os.path.join(raw_dir, MNIST_FILES[1][0]), 8)
    test_x = _parse_idx(os.path.join(raw_dir, MNIST_FILES[2][0]), 16) / 255.0
    test_y = _parse_idx(os.path.join(raw_dir, MNIST_FILES[3][0]), 8)
    for (_, out), arr in zip(MNIST_FILES, [train_x, train_y, test_x, test_y]):
        np.save(os.path.join(npy_dir, out), arr)
    return (torch.from_numpy(train_x).float(), torch.from_numpy(train_y).long(),
            torch.from_numpy(test_x).float(), torch.from_numpy(test_y).long())


def make_synthetic_mnist(n, seed=0, hw=28):
    """Offline fallback: blob images that still exercise the whole training loop."""
    rng = np.random.default_rng(seed)
    imgs = np.zeros((n, hw * hw), dtype=np.float32)
    labels = np.full(n, -1, dtype=np.int64)
    yy, xx = np.meshgrid(np.arange(hw), np.arange(hw))
    for i in range(n):
        d = i % 10
        labels[i] = d
        cx = (d % 5) * 5.0 + 4.0
        cy = (d // 5) * 5.0 + 4.0
        im = np.zeros((hw, hw), dtype=np.float32)
        for _ in range(int(rng.integers(2, 5))):
            ang = rng.uniform(0, np.pi)
            r = rng.uniform(0.5, 3.0)
            ox, oy = cx + r * np.cos(ang), cy + r * np.sin(ang)
            im += np.exp(-(((xx - ox) / 2.2) ** 2 + ((yy - oy) / 2.2) ** 2)) * rng.uniform(0.4, 1.0)
        imgs[i] = np.clip(im, 0, 1).reshape(-1)
    log(f"[data] synthetic MNIST-like blobs: {n} x {hw}x{hw}")
    return torch.from_numpy(imgs).float(), torch.from_numpy(labels).long()


# Model (same architecture as models.VAE, so the checkpoints are interchangeable)
LOGVAR_MIN, LOGVAR_MAX = -6.0, 4.0     # log-variance clip: std in [0.05, 7.4]


def _gn(ch, max_groups=8):
    """GroupNorm with a group count that divides `ch` (robust to small bases)."""
    g = min(max_groups, ch)
    while ch % g:
        g -= 1
    return nn.GroupNorm(g, ch)


class VAE(nn.Module):
    """Conv VAE: x (1x28x28, [-1,1]) <-> z (C x 7 x 7).

    Encoder  : two stride-2 convs (28 -> 14 -> 7) + GroupNorm/SiLU, 1x1-conv heads
    Decoder  : conv + two stride-2 transposed convs (7 -> 14 -> 28), tanh output
    The tanh is applied inside `decode()` as a parameterless module, so the
    saved state_dict holds no extra entries.
    """

    def __init__(self, in_channels=1, latent_channels=8, downsample=4, base=32):
        super().__init__()
        assert downsample in (2, 4), "downsample must be 2 or 4"
        n_stages = 1 if downsample == 2 else 2
        self.downsample = downsample

        self.enc = nn.ModuleList()
        cin, ch = in_channels, base
        for _ in range(n_stages):
            self.enc.append(nn.Sequential(
                nn.Conv2d(cin, ch, 4, stride=2, padding=1),
                _gn(ch), nn.SiLU(),
            ))
            cin, ch = ch, ch * 2
        enc_out = base * (2 ** (n_stages - 1))
        self.mu = nn.Conv2d(enc_out, latent_channels, 3, padding=1)
        self.logvar = nn.Conv2d(enc_out, latent_channels, 3, padding=1)
        # small initial variance: training starts near-deterministic
        nn.init.zeros_(self.logvar.weight)
        nn.init.constant_(self.logvar.bias, -2.0)

        self.latent_channels = latent_channels
        ch = base * (2 ** n_stages)
        dec = [nn.Sequential(nn.Conv2d(latent_channels, ch, 3, padding=1),
                             _gn(ch), nn.SiLU())]
        for s in reversed(range(n_stages)):
            out_ch = base * (2 ** (s + 1)) if s > 0 else base
            dec.append(nn.Sequential(
                nn.ConvTranspose2d(ch, out_ch, 4, stride=2, padding=1),
                _gn(out_ch), nn.SiLU(),
            ))
            ch = out_ch
        dec.append(nn.Conv2d(ch, in_channels, 3, padding=1))
        self.dec = nn.Sequential(*dec)
        self.final_act = nn.Tanh()          # no params -> state_dict unchanged

    def encode(self, x):
        h = x
        for layer in self.enc:
            h = layer(h)
        mu = self.mu(h)
        logvar = self.logvar(h).clamp(LOGVAR_MIN, LOGVAR_MAX)
        return mu, logvar

    def reparameterize(self, mu, logvar, sample=True):
        if not sample:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z):
        """z -> image in [-1, 1]."""
        return self.final_act(self.dec(z))

    def forward(self, x, sample=True):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar, sample=sample)
        recon = self.decode(z)
        return recon, mu, logvar, z


def vae_loss(recon, x, mu, logvar, beta=1.0, loss_type="bce", free_bits=0.0):
    """Return (total, rec_nats, kl_nats, kl_per_dim).

    All reconstruction terms are *summed over pixels per sample* and the KL is
    *summed over latent dims per sample*, so both are in nats and beta=1 is the
    textbook VAE objective.  kl_per_dim is the mean nats per latent dimension.
    """
    n = x.shape[0]
    t = (x + 1.0) / 2.0                                   # [-1,1] -> [0,1]
    p = ((recon + 1.0) / 2.0).clamp(1e-6, 1.0 - 1e-6)     # tanh   -> [0,1]
    if loss_type == "bce":
        rec = F.binary_cross_entropy(p, t, reduction="sum") / n
    else:
        rec = F.mse_loss(recon, x, reduction="sum") / n

    kl_dims = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())   # (N, C, H, W)
    kl_per_dim_map = kl_dims.mean(dim=0)
    if free_bits > 0:
        kl_per_dim_map = kl_per_dim_map.clamp(min=free_bits)
    kl = kl_per_dim_map.sum()                              # nats per sample
    kl_per_dim = kl_per_dim_map.mean().item()
    return rec + beta * kl, rec, kl, kl_per_dim


#
# Train / evaluate
#
@torch.no_grad()
def evaluate(vae, x, device, batch=512, loss_type="bce", beta=1.0):
    """Deterministic (mu) evaluation on pixels, in both [-1,1] and [0,1] units."""
    vae.eval()
    n = len(x)
    se_c = 0.0        # squared error on [-1,1]
    se_01 = 0.0       # squared error on [0,1]
    rec_sum = kl_sum = 0.0
    mus = []
    for i in range(0, n, batch):
        xb = x[i:i + batch].to(device)
        mu, logvar = vae.encode(xb)
        recon = vae.decode(mu)                       # deterministic path (as pipeline)
        _, rec, kl, _ = vae_loss(recon, xb, mu, logvar, beta=beta, loss_type=loss_type)
        rec_sum += rec.item() * len(xb)
        kl_sum += kl.item() * len(xb)
        se_c += F.mse_loss(recon, xb, reduction="sum").item()
        se_01 += F.mse_loss((recon + 1) / 2, (xb + 1) / 2, reduction="sum").item()
        mus.append(mu.reshape(len(xb), -1).cpu())
    mu_all = torch.cat(mus, dim=0)
    mse_c = se_c / (n * x.shape[1] * x.shape[2] * x.shape[3])
    mse_01 = se_01 / (n * x.shape[1] * x.shape[2] * x.shape[3])
    psnr = 10.0 * np.log10(1.0 / max(mse_01, 1e-12))
    return {
        "recon_mse_centered": mse_c,          # same metric the flow pipeline prints
        "recon_mse_01": mse_01,
        "psnr_db": psnr,
        "rec_nats": rec_sum / n,
        "kl_nats": kl_sum / n,
        "kl_per_dim": kl_sum / n / mu_all.shape[1],
        "mu_mean": mu_all.mean().item(),
        "mu_std": mu_all.std().item(),
        "mu_dim_std_mean": mu_all.std(dim=0).mean().item(),
        "mu_dim_std_max": mu_all.std(dim=0).max().item(),
        "n_dims_active": int((mu_all.std(dim=0) > 0.1).sum().item()),
        "n_latent_dims": int(mu_all.shape[1]),
    }


def train(vae, train_x, test_x, args, device, out_dir):
    opt = torch.optim.Adam(vae.parameters(), lr=args.lr)
    lat_h = 28 // vae.downsample
    total_steps = args.epochs * args.steps
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=total_steps, eta_min=args.lr_min)
    warm = max(1, args.kl_warmup_steps)
    hist = []
    t0 = time.time()
    step = 0
    # Best-checkpoint tracking: the KL term keeps trading reconstruction for a
    # smaller KL after the decoder has converged, so test reconstruction peaks
    # early and then decays. The saved weights are the ones at the best test MSE.
    best = {"mse": float("inf"), "epoch": -1, "state": None}
    bad_epochs = 0
    log(f"\n[train] {args.epochs} epochs x {args.steps} steps, batch {args.batch}, "
        f"lr {args.lr}->{args.lr_min}")
    log(f"[train] loss = rec_nats + beta*kl_nats ; beta ramp 0 -> {args.beta} "
        f"over {warm} steps (both terms in nats)")
    log(f"[train] latent {vae.latent_channels}x{lat_h}x{lat_h} = "
        f"{vae.latent_channels * lat_h * lat_h} dims (vs {28 * 28} pixels)")
    log(f"[train] keeping BEST checkpoint by TEST recon MSE "
        f"(patience {args.patience} epochs, 0=off)")

    for ep in range(args.epochs):
        vae.train()
        agg = {"loss": 0.0, "rec": 0.0, "kl": 0.0}
        for it in range(args.steps):
            beta = args.beta * min(1.0, step / warm)
            idx = torch.randint(0, len(train_x), (args.batch,))
            xb = train_x[idx].to(device)
            recon, mu, logvar, _ = vae(xb)
            loss, rec, kl, kl_dim = vae_loss(
                recon, xb, mu, logvar, beta=beta,
                loss_type=args.loss, free_bits=args.free_bits)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(vae.parameters(), args.grad_clip)
            opt.step()
            sched.step()
            step += 1
            agg["loss"] += loss.item()
            agg["rec"] += rec.item()
            agg["kl"] += kl.item()
            if args.log_every and it % args.log_every == 0:
                log(f"  ep {ep+1:2d}/{args.epochs} it {it:4d} beta {beta:.3f} "
                    f"loss {loss.item():7.2f} (rec {rec.item():7.2f} kl {kl.item():7.2f} "
                    f"kl/dim {kl_dim:.3f})")
        for k in agg:
            agg[k] /= args.steps
        ev = evaluate(vae, test_x, device, loss_type=args.loss, beta=args.beta)
        ev.update({"epoch": ep + 1, "beta": args.beta, "train": dict(agg),
                   "lr": sched.get_last_lr()[0], "elapsed_s": time.time() - t0})
        hist.append(ev)
        improved = ev["recon_mse_01"] < best["mse"]
        if improved:
            best.update({"mse": ev["recon_mse_01"], "epoch": ep + 1,
                         "state": {k: v.detach().clone() for k, v in vae.state_dict().items()}})
            bad_epochs = 0
        else:
            bad_epochs += 1
        log(f"[ep {ep+1:2d}/{args.epochs}] train rec {agg['rec']:7.2f} kl {agg['kl']:7.2f} | "
            f"TEST mse[-1,1] {ev['recon_mse_centered']:.5f} mse[0,1] {ev['recon_mse_01']:.5f} "
            f"PSNR {ev['psnr_db']:.2f}dB | mu_std {ev['mu_std']:.3f} "
            f"active-dims {ev['n_dims_active']}/{ev['n_latent_dims']} ({ev['elapsed_s']:.0f}s)"
            + ("   best so far" if improved else f"   (no gain for {bad_epochs})"))
        if args.viz_every and (ep + 1) % args.viz_every == 0:
            save_recon_grid(vae, test_x, device, os.path.join(out_dir, "recon_grid.png"),
                            n=16, title=f"epoch {ep+1}")
            plot_loss_curve(hist, os.path.join(out_dir, "loss_curve.png"))
        if args.patience and bad_epochs >= args.patience:
            log(f"[train] early stop: no TEST improvement for {bad_epochs} epochs "
                f"(best = epoch {best['epoch']}, mse[0,1] {best['mse']:.5f})")
            break

    # restore the best weights (NOT the last) into the model
    if best["state"] is not None:
        vae.load_state_dict(best["state"])
        log(f"[train] restored BEST weights from epoch {best['epoch']} "
            f"(mse[0,1] {best['mse']:.5f})")
    return hist, best["epoch"]


#
# Visualization
#
@torch.no_grad()
def _decode(vae, z):
    vae.eval()
    return vae.decode(z)


def _to01(x):
    return ((x.clamp(-1, 1) + 1.0) / 2.0).cpu().numpy()


def save_recon_grid(vae, x, device, path, n=16, title=""):
    """Top row: real digits. Bottom row: their reconstructions from mu."""
    n = min(n, len(x))
    xb = x[:n].to(device)
    mu, _ = vae.encode(xb)
    rec = _decode(vae, mu)
    real01, rec01 = _to01(xb), _to01(rec)
    fig, axes = plt.subplots(2, n, figsize=(1.4 * n, 3.2))
    for i in range(n):
        axes[0, i].imshow(real01[i, 0], cmap="gray", vmin=0, vmax=1)
        axes[1, i].imshow(rec01[i, 0], cmap="gray", vmin=0, vmax=1)
        axes[0, i].axis("off"); axes[1, i].axis("off")
    axes[0, 0].set_ylabel("REAL", fontsize=9)
    axes[1, 0].set_ylabel("RECON", fontsize=9)
    plt.suptitle(f"VAE reconstruction {title}".strip())
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


@torch.no_grad()
def save_samples_grid(vae, latent_shape, device, path, n=32, seed=0):
    """Prior samples: z ~ N(0, I), decoded."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    z = torch.randn(n, *latent_shape, generator=g).to(device)
    imgs = _to01(_decode(vae, z))
    cols = min(n, 8)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(1.4 * cols, 1.4 * rows))
    axes = np.atleast_2d(axes)
    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        ax.axis("off")
        if i < n:
            ax.imshow(imgs[i, 0], cmap="gray", vmin=0, vmax=1)
    plt.suptitle("z ~ N(0,I) -> VAE decoder")
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


def plot_loss_curve(hist, path):
    if not hist:
        return
    ep = [h["epoch"] for h in hist]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    axes[0].plot(ep, [h["train"]["rec"] for h in hist], "-o", label="rec (nats)")
    axes[0].plot(ep, [h["train"]["kl"] for h in hist], "-s", label="kl (nats)")
    axes[0].set_yscale("log"); axes[0].set_xlabel("epoch"); axes[0].legend()
    axes[0].set_title("train loss terms (nats, balanced)")
    axes[1].plot(ep, [h["recon_mse_centered"] for h in hist], "-o")
    axes[1].set_xlabel("epoch"); axes[1].set_title("TEST recon MSE ([-1,1] scale)")
    axes[1].set_yscale("log")
    axes[2].plot(ep, [h["psnr_db"] for h in hist], "-o", color="tab:green")
    axes[2].set_xlabel("epoch"); axes[2].set_title("TEST PSNR (dB)")
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


@torch.no_grad()
def save_latent_stats(vae, x, device, path, batch=512):
    """Latent diagnostics that matter for the flow: mu close to N(0,1), with
    most dimensions actually used (active dims >> 0)."""
    mus = []
    for i in range(0, len(x), batch):
        mu, _ = vae.encode(x[i:i + batch].to(device))
        mus.append(mu.reshape(len(mu), -1).cpu())
    mu = torch.cat(mus, 0).numpy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    axes[0].hist(mu.reshape(-1), bins=80, density=True, alpha=0.8)
    xs = np.linspace(-4, 4, 200)
    axes[0].plot(xs, np.exp(-xs ** 2 / 2) / np.sqrt(2 * np.pi), "r-", lw=1.5,
                 label="N(0,1)")
    axes[0].set_title("posterior mu values"); axes[0].legend()
    axes[1].bar(np.arange(mu.shape[1]), mu.std(0), width=1.0)
    axes[1].set_title(f"per-dim std of mu  (active>0.1: {(mu.std(0) > 0.1).sum()}/{mu.shape[1]})")
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


def verdict(ev):
    m = ev["recon_mse_01"]
    if m < 0.005:
        q = "excellent: clean target for the latent FM"
    elif m < 0.015:
        q = "good: digits clearly recognisable"
    elif m < 0.04:
        q = "weak: digits visible but blurry, train longer"
    else:
        q = "bad: the VAE is not reconstructing"
    return q


#
# Main
#
def build_args():
    ap = argparse.ArgumentParser(
        description="VAE trainer for the latent flow-matching pipeline")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="vae_run")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use-synthetic", action="store_true",
                    help="skip MNIST and train on blob images (offline smoke test)")

    # architecture (must match models.VAE so the checkpoint is reusable)
    ap.add_argument("--latent-channels", type=int, default=8)
    ap.add_argument("--downsample", type=int, default=4, choices=[2, 4], help="28 -> 7 (4) or 14 (2)")
    ap.add_argument("--base", type=int, default=32)

    # optimisation
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--steps", type=int, default=150, help="steps per epoch")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-min", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=10.0)

    # loss
    ap.add_argument("--loss", choices=["bce", "mse"], default="bce")
    ap.add_argument("--beta", type=float, default=1.0,
                    help="weight on the SUMMED KL (nats). 1.0 = standard VAE")
    ap.add_argument("--kl-warmup-steps", type=int, default=1500,
                    help="ramp beta 0 -> beta over this many steps")
    ap.add_argument("--free-bits", type=float, default=0.0,
                    help="per-dim KL floor in nats (encourages using all dims)")

    # logging / io
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--viz-every", type=int, default=5)
    ap.add_argument("--patience", type=int, default=8,
                    help="early stop after N epochs without TEST recon improvement (0 = off)")
    ap.add_argument("--train-size", type=int, default=0, help="0 = all 60000")
    ap.add_argument("--eval-size", type=int, default=2000)
    ap.add_argument("--sanity", action="store_true", help="tiny CPU run to check the plumbing")
    return ap


def sanity_overrides(args):
    args.epochs, args.steps, args.batch = 3, 30, 64
    args.base, args.latent_channels = 16, 4
    args.train_size, args.eval_size = 2000, 256
    args.kl_warmup_steps, args.log_every, args.viz_every = 30, 10, 1
    args.out_dir = "vae_run_sanity"
    return args


def main():
    args = build_args().parse_args()
    if args.sanity:
        args = sanity_overrides(args)
    device = torch.device(args.device if args.device != "auto"
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    log("=" * 78)
    log("VAE TRAINING")
    log("=" * 78)
    log(f"[device] {device}   torch {torch.__version__}")
    log(f"[args]   {json.dumps(vars(args), indent=2, default=str)}")

    # ---------------- data ----------------
    try:
        if args.use_synthetic:
            raise RuntimeError("forced synthetic")
        train_x, _, test_x, _ = load_mnist(args.data_dir)
    except Exception as e:  # noqa: BLE001
        log(f"[data] real MNIST unavailable ({e}); using synthetic blobs")
        train_x, _ = make_synthetic_mnist(5000, seed=args.seed)
        test_x, _ = make_synthetic_mnist(2000, seed=args.seed + 1)

    if args.train_size and args.train_size < len(train_x):
        sub = torch.randperm(len(train_x))[:args.train_size]
        train_x = train_x[sub]
    if args.eval_size and args.eval_size < len(test_x):
        sub = torch.randperm(len(test_x))[:args.eval_size]
        test_x = test_x[sub]
    train_x = train_x.reshape(-1, 1, 28, 28).float() * 2.0 - 1.0   # [-1,1]
    test_x = test_x.reshape(-1, 1, 28, 28).float() * 2.0 - 1.0
    log(f"[data] train {tuple(train_x.shape)}  test {tuple(test_x.shape)}  "
        f"range [{train_x.min():.2f},{train_x.max():.2f}] mean {train_x.mean():.3f} "
        f"std {train_x.std():.3f}")

    # ---------------- model ----------------
    vae = VAE(in_channels=1, latent_channels=args.latent_channels,
              downsample=args.downsample, base=args.base).to(device)
    n_par = sum(p.numel() for p in vae.parameters())
    log(f"[model] VAE params {n_par/1e3:.1f}k, latent "
        f"{args.latent_channels}x{28//args.downsample}x{28//args.downsample}")

    hist, best_epoch = train(vae, train_x, test_x, args, device, args.out_dir)
    final = hist[-1]
    best = hist[best_epoch - 1] if best_epoch > 0 else final

    # ---------------- final artifacts ----------------
    vae.eval()
    torch.save(vae.state_dict(), os.path.join(args.out_dir, "vae.pt"))
    latent_shape = (args.latent_channels, 28 // args.downsample, 28 // args.downsample)
    save_recon_grid(vae, test_x, device, os.path.join(args.out_dir, "recon_grid.png"),
                    n=16, title="(final)")
    save_samples_grid(vae, latent_shape, device, os.path.join(args.out_dir, "samples_grid.png"))
    plot_loss_curve(hist, os.path.join(args.out_dir, "loss_curve.png"))
    save_latent_stats(vae, test_x, device, os.path.join(args.out_dir, "latent_stats.png"))
    meta = {
        "args": vars(args),
        "arch": {"in_channels": 1, "latent_channels": args.latent_channels,
                 "downsample": args.downsample, "base": args.base},
        "latent_shape": list(latent_shape),
        "final_metrics": final,
        "best_epoch": best_epoch,
        "best_metrics": best,
        "history": hist,
    }
    with open(os.path.join(args.out_dir, "vae_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)

    log("\n" + "=" * 78)
    log("VAE TRAINING RESULT")
    log("=" * 78)
    log(f"  SAVED WEIGHTS = BEST epoch {best_epoch} (not the last epoch)")
    log(f"  recon MSE ([-1,1] scale)  {best['recon_mse_centered']:.5f}"
        "   (the metric the FM pipeline reports)")
    log(f"  recon MSE ([0,1] scale)   {best['recon_mse_01']:.5f}")
    log(f"  PSNR                      {best['psnr_db']:.2f} dB")
    log(f"  rec / kl (nats)           {best['rec_nats']:.1f} / {best['kl_nats']:.1f}"
        f"   (kl/dim {best['kl_per_dim']:.3f})")
    log(f"  latent mu: mean {best['mu_mean']:+.3f}  std {best['mu_std']:.3f}  "
        f"active-dims {best['n_dims_active']}/{best['n_latent_dims']}")
    log(f"  VERDICT: {verdict(best)}")
    log(f"\n  artifacts -> {os.path.abspath(args.out_dir)}")
    log("    vae.pt (use with: python train_fm.py --vae-checkpoint vae_run/vae.pt)")
    log("    recon_grid.png  (top = real, bottom = reconstruction)")
    log("=" * 78)


if __name__ == "__main__":
    main()
