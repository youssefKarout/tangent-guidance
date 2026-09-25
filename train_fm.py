"""Stage-A trainer: flow matching from noise to frozen VAE latents (MNIST).

Trains only the velocity field on latents produced by a frozen VAE:

    z ~ q_vae(x) (frozen encoder mean) ; x0 ~ N(0, I)
    x_t = (1 - t) * x0 + t * z (+ small Gaussian noise)
    target v = z - x0

Design choices relative to a plain conv flow:
  * logit-normal t sampling by default, which puts more weight on the middle
    of the path where the velocity target is hardest,
  * sinusoidal time embedding with FiLM modulation in every residual block,
  * residual conv stack with GroupNorm + SiLU, AdamW with cosine schedule,
  * EMA weights used for sampling, optional Heun (2nd-order) sampler,
  * per-t-bucket velocity MSE reported next to the loss.

Usage:
    python train_fm.py --vae-checkpoint vae_run/vae.pt
    python train_fm.py --sanity --use-synthetic
    python train_fm.py --device cuda --vae-checkpoint vae_run/vae.pt --epochs 30 --batch 512

Outputs in --out-dir: fm.pt, fm_ema.pt, fm_meta.json, loss_curve.png,
samples_grid.png and tbucket_curve.png.
"""

import os
import json
import time
import argparse
import gzip
import struct
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Data helpers
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
    log(f"    downloading {url} -> {dest}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as f:
        f.write(resp.read())


def _parse_idx(gz_path, kind):
    """Parse IDX gzip. kind='images' -> (N,784) uint8; kind='labels' -> (N,) int64."""
    with gzip.open(gz_path, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        n = struct.unpack(">I", f.read(4))[0]
        if kind == "images":
            if magic != 2051:
                raise ValueError(f"bad magic {magic} in {gz_path}")
            rows = struct.unpack(">I", f.read(4))[0]
            cols = struct.unpack(">I", f.read(4))[0]
            buf = f.read()
            arr = np.frombuffer(buf, dtype=np.uint8)
            need = n * rows * cols
            if arr.size != need:
                raise ValueError(
                    f"truncated {gz_path}: got {arr.size} px, need {need} "
                    f"(n={n} {rows}x{cols})")
            return arr.reshape(n, rows * cols)
        if magic != 2049:
            raise ValueError(f"bad magic {magic} in {gz_path}")
        buf = f.read()
        arr = np.frombuffer(buf, dtype=np.uint8)
        if arr.size != n:
            raise ValueError(f"truncated {gz_path}: got {arr.size}, need {n}")
        return arr.astype(np.int64)


def load_mnist(data_dir, offline=False):
    raw_dir = os.path.join(data_dir, "mnist", "raw")
    npy_dir = os.path.join(data_dir, "mnist")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(npy_dir, exist_ok=True)
    paths = [os.path.join(npy_dir, out) for _, out in MNIST_FILES]
    if all(os.path.exists(p) for p in paths):
        log(f"[data] cached MNIST from {npy_dir}")
        return (torch.from_numpy(np.load(paths[0])).float(),
                torch.from_numpy(np.load(paths[1])).long(),
                torch.from_numpy(np.load(paths[2])).float(),
                torch.from_numpy(np.load(paths[3])).long())
    if offline:
        raise RuntimeError("MNIST cache missing and --offline set")
    # Decoded length = header + payload. A truncated download can still open as
    # gzip, so the expected size is checked here and the file is re-downloaded
    # in the same run instead of failing later in _parse_idx.
    _EXPECTED = {
        "train-images-idx3-ubyte.gz": 16 + 60000 * 784,
        "train-labels-idx1-ubyte.gz": 8 + 60000,
        "t10k-images-idx3-ubyte.gz": 16 + 10000 * 784,
        "t10k-labels-idx1-ubyte.gz": 8 + 10000,
    }
    for fname, _ in MNIST_FILES:
        gz = os.path.join(raw_dir, fname)
        if os.path.exists(gz):
            try:
                with gzip.open(gz, "rb") as vf:
                    data = vf.read()
                if len(data) != _EXPECTED[fname]:
                    raise ValueError(f"size {len(data)} != expected {_EXPECTED[fname]}")
            except Exception as e:
                log(f"    corrupt/incomplete {gz} ({e}), re-downloading")
                os.remove(gz)
        if os.path.exists(gz):
            continue
        last = None
        for base in MNIST_MIRRORS:
            try:
                _download(base + fname, gz)
                if os.path.getsize(gz) == 0:
                    os.remove(gz)
                    raise RuntimeError("downloaded 0 bytes")
                # downloaded file must open as gzip and carry a valid IDX magic
                with gzip.open(gz, "rb") as vf:
                    magic = struct.unpack(">I", vf.read(4))[0]
                if magic not in (2051, 2049):
                    os.remove(gz)
                    raise RuntimeError(f"bad IDX magic {magic}")
                break
            except Exception as e:  # noqa: BLE001
                last = e
                if os.path.exists(gz):
                    try:
                        os.remove(gz)
                    except OSError:
                        pass
        else:
            raise RuntimeError(f"failed to download {fname}: {last}")
    try:
        tx = _parse_idx(os.path.join(raw_dir, MNIST_FILES[0][0]), "images") / 255.0
        ty = _parse_idx(os.path.join(raw_dir, MNIST_FILES[1][0]), "labels")
        ex = _parse_idx(os.path.join(raw_dir, MNIST_FILES[2][0]), "images") / 255.0
        ey = _parse_idx(os.path.join(raw_dir, MNIST_FILES[3][0]), "labels")
    except Exception as e:
        # raw files on disk can be incomplete (e.g. an interrupted download);
        # clear them so the next run fetches fresh copies.
        log(f"    parse failed ({e}); deleting raw cache for clean re-download")
        for fname, _ in MNIST_FILES:
            gz = os.path.join(raw_dir, fname)
            try:
                if os.path.exists(gz):
                    os.remove(gz)
            except OSError:
                pass
        raise RuntimeError(f"MNIST raw cache was corrupt and has been cleared: {e}. "
                           "Re-run the same command to re-download.") from e
    for (_, out), arr in zip(MNIST_FILES, [tx, ty, ex, ey]):
        np.save(os.path.join(npy_dir, out), arr)
    return (torch.from_numpy(tx).float(), torch.from_numpy(ty).long(),
            torch.from_numpy(ex).float(), torch.from_numpy(ey).long())


def make_synthetic_mnist(n, seed=0, hw=28):
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
            im += np.exp(-(((xx - cx - r * np.cos(ang)) / 2.2) ** 2
                            + ((yy - cy - r * np.sin(ang)) / 2.2) ** 2)) * rng.uniform(0.4, 1.0)
        imgs[i] = np.clip(im, 0, 1).reshape(-1)
    return torch.from_numpy(imgs).float(), torch.from_numpy(labels).long()

# Frozen VAE (same architecture as train_vae.VAE)
LOGVAR_MIN, LOGVAR_MAX = -6.0, 4.0


def _gn(ch, max_groups=8):
    g = min(max_groups, ch)
    while ch % g:
        g -= 1
    return nn.GroupNorm(g, ch)


class VAE(nn.Module):
    def __init__(self, in_channels=1, latent_channels=8, downsample=4, base=32):
        super().__init__()
        assert downsample in (2, 4)
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
        self.final_act = nn.Tanh()

    def encode(self, x):
        h = x
        for layer in self.enc:
            h = layer(h)
        return self.mu(h), self.logvar(h).clamp(LOGVAR_MIN, LOGVAR_MAX)

    def decode(self, z):
        return self.final_act(self.dec(z))


@torch.no_grad()
def encode_data(vae, x, device, batch=512):
    vae.eval()
    outs = []
    for i in range(0, len(x), batch):
        outs.append(vae.encode(x[i:i + batch].to(device))[0].cpu())
    return torch.cat(outs, dim=0)


@torch.no_grad()
def decode_data(vae, z, batch=512):
    vae.eval()
    dev = next(vae.parameters()).device
    outs = []
    z = z.to(dev)
    for i in range(0, len(z), batch):
        outs.append(vae.decode(z[i:i + batch]).cpu())
    return torch.cat(outs, dim=0)

# Flow-matching network: residual conv stack, sinusoidal t, FiLM
class TimeEmbed(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim * 4))

    def forward(self, t):
        t = t.reshape(-1)
        half = self.dim // 2
        freqs = torch.exp(-np.log(10000.0) * torch.arange(half, device=t.device) / half)
        args = t[:, None] * freqs[None, :]
        return self.mlp(torch.cat([args.sin(), args.cos()], dim=1))


class ResBlock(nn.Module):
    def __init__(self, ch, tdim):
        super().__init__()
        self.n1 = _gn(ch)
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.n2 = _gn(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.film = nn.Linear(tdim, ch * 2)
        nn.init.zeros_(self.c2.weight); nn.init.zeros_(self.c2.bias)

    def forward(self, h, te):
        s, b = self.film(te).chunk(2, dim=1)
        s = s[:, :, None, None]; b = b[:, :, None, None]
        h0 = h
        h = self.c1(F.silu(self.n1(h)))
        h = self.n2(h) * (1 + s) + b
        h = self.c2(F.silu(h))
        return h0 + h


class FM_Conv(nn.Module):
    def __init__(self, latent_shape=(8, 7, 7), base=64, depth=4, tdim=128):
        super().__init__()
        c = latent_shape[0]
        self.inp = nn.Conv2d(c, base, 3, padding=1)
        self.temb = TimeEmbed(tdim)
        self.blocks = nn.ModuleList([ResBlock(base, tdim * 4) for _ in range(depth)])
        self.out = nn.Sequential(_gn(base), nn.SiLU(), nn.Conv2d(base, c, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight); nn.init.zeros_(self.out[-1].bias)

    def forward(self, x, t):
        te = self.temb(t.reshape(-1))
        h = self.inp(x)
        for blk in self.blocks:
            h = blk(h, te)
        return self.out(h)

# Flow-matching objective: t sampling, path noise, target-norm weighting
def sample_t(n, device, mode="logitnorm", uniform_eps=1e-3):
    if mode == "uniform":
        return torch.rand(n, 1, device=device).clamp(uniform_eps, 1 - uniform_eps)
    if mode == "mid":
        t = torch.rand(n, 1, device=device)
        return 0.15 + 0.7 * t
    u = torch.randn(n, 1, device=device)
    return torch.sigmoid(u).clamp(1e-3, 1 - 1e-3)


def fm_step(fm, z1, device, t_mode="logitnorm", sigma=0.0, w_clip=5.0):
    zb = z1[torch.randint(0, len(z1), (z1.shape[0],))].to(device)
    x0 = torch.randn_like(zb)
    t = sample_t(len(zb), device, t_mode)
    tt = t.reshape(-1, 1, 1, 1)
    xt = (1 - tt) * x0 + tt * zb
    if sigma and sigma > 0:
        xt = xt + sigma * torch.randn_like(xt)
    v_pred = fm(xt, t)
    v_tgt = zb - x0
    err = v_pred - v_tgt
    mse = err.pow(2).mean(dim=[1, 2, 3])
    tgt = v_tgt.pow(2).mean(dim=[1, 2, 3]).clamp_min(1e-8)
    w = (tgt.mean() / tgt).clamp(1.0 / w_clip, w_clip)
    return (mse * w).mean(), mse.detach(), t.detach()


class EMA:
    def __init__(self, model, decay=0.999):
        import copy
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        for e, m in zip(self.ema.parameters(), model.parameters()):
            e.mul_(self.decay).add_(m.detach(), alpha=1 - self.decay)

# Samplers (Euler and Heun)
@torch.no_grad()
def sample_fm(model, n, steps, batch, device, shape, sampler="euler"):
    out = []
    for i in range(0, n, batch):
        b = min(batch, n - i)
        x = torch.randn(b, *shape, device=device)
        dt = 1.0 / steps
        for j in range(steps):
            t = torch.full((b, 1), j * dt, device=device)
            v = model(x, t)
            if sampler == "heun" and j < steps - 1:
                t2 = torch.full((b, 1), (j + 1) * dt, device=device)
                x_star = x + dt * v
                x = x + 0.5 * dt * (v + model(x_star, t2))
            else:
                x = x + dt * v
        out.append(x.cpu())
    return torch.cat(out, dim=0)


@torch.no_grad()
def eval_mse(fm, train_z, eval_z, device, batch=512, n=2000, t_mode="logitnorm"):
    fm.eval()
    bu = []
    for split in (train_z, eval_z):
        ms = []
        for _ in range(max(1, n // batch)):
            idx = torch.randint(0, len(split), (batch,))
            zb = split[idx].to(device)
            x0 = torch.randn_like(zb)
            t = sample_t(batch, device, t_mode)
            xt = (1 - t.reshape(-1, 1, 1, 1)) * x0 + t.reshape(-1, 1, 1, 1) * zb
            ms.append(F.mse_loss(fm(xt, t), zb - x0).item())
        bu.append(float(np.mean(ms)))
    fm.train()
    return bu[0], bu[1]

# Per-t diagnostics and plots
@torch.no_grad()
def tbucket_report(fm, z, device, n_bins=10, batch=256, n=2000):
    fm.eval()
    edges = np.linspace(0, 1, n_bins + 1)
    acc = [[] for _ in range(n_bins)]
    for _ in range(max(1, n // batch)):
        idx = torch.randint(0, len(z), (batch,))
        zb = z[idx].to(device)
        x0 = torch.randn_like(zb)
        t = torch.rand(batch, 1, device=device)
        xt = (1 - t.reshape(-1, 1, 1, 1)) * x0 + t.reshape(-1, 1, 1, 1) * zb
        e = (fm(xt, t) - (zb - x0)).pow(2).mean(dim=[1, 2, 3]).cpu().numpy()
        b = np.clip((t.cpu().numpy().ravel() * n_bins).astype(int), 0, n_bins - 1)
        for k in range(batch):
            acc[b[k]].append(e[k])
    fm.train()
    return edges, np.array([float(np.mean(a)) if a else float("nan") for a in acc])


def montage(x, path, n=64, title=""):
    x = x[:n]
    v = x.clamp(-1, 1)
    img = (v + 1) / 2
    n = len(img)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.1, rows * 1.1))
    axes = np.atleast_2d(axes)
    for k in range(rows * cols):
        ax = axes[k // cols, k % cols]
        ax.axis("off")
        if k < n:
            im = img[k, 0].cpu().numpy()
            ax.imshow(im, cmap="gray", vmin=0, vmax=1)
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


def plot_scores(hist, path):
    ep = [h["epoch"] for h in hist]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, [h["train_mse"] for h in hist], label="train")
    ax[0].plot(ep, [h["test_mse"] for h in hist], label="test")
    ax[0].set_yscale("log"); ax[0].legend(); ax[0].grid(True, alpha=0.3)
    ax[0].set_title("FM velocity MSE (log)")
    ax[1].plot(ep, [h["gen_px_mse"] for h in hist], label="generated px MSE")
    ax[1].axhline(hist[0]["vae_ceil_mse"], ls="--", c="k", label="VAE ceiling")
    ax[1].legend(); ax[1].grid(True, alpha=0.3)
    ax[1].set_title("decoded generation quality")
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()


def plot_tbuckets(edges, curves, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    for tag, vals in curves:
        ax.plot((edges[:-1] + edges[1:]) / 2, vals, marker="o", label=tag)
    ax.set_yscale("log"); ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_xlabel("t"); ax.set_ylabel("velocity MSE")
    ax.set_title("velocity MSE by t bucket")
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()

# CLI
def build_parser():
    p = argparse.ArgumentParser(description="Stage-A FM on frozen VAE latents")
    p.add_argument("--vae-checkpoint", type=str, required=True)
    p.add_argument("--latent-channels", type=int, default=8)
    p.add_argument("--vae-base", type=int, default=32)
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--out-dir", type=str, default="fm_run")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--use-synthetic", action="store_true")
    p.add_argument("--sanity", action="store_true")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-min", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--base", type=int, default=64)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--tdim", type=int, default=128)
    p.add_argument("--t-sampling", type=str, default="logitnorm",
                   choices=["logitnorm", "uniform", "mid"])
    p.add_argument("--sigma", type=float, default=0.0)
    p.add_argument("--w-clip", type=float, default=5.0)
    p.add_argument("--sample-steps", type=int, default=50)
    p.add_argument("--sampler", type=str, default="euler", choices=["euler", "heun"])
    p.add_argument("--viz-every", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    return p

# Argument handling and setup
def run_training(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()
    if args.sanity:
        args.epochs = 3; args.steps = 10; args.batch = 64
        args.base = 16; args.depth = 2
        args.sample_steps = 10; args.viz_every = 1

    if args.use_synthetic:
        train_x, train_y = make_synthetic_mnist(8000)
        test_x, test_y = make_synthetic_mnist(1500, seed=1)
    else:
        train_x, train_y, test_x, test_y = load_mnist(args.data_dir, args.offline)
    log(f"[data] train {tuple(train_x.shape)} test {tuple(test_x.shape)}")
    train_p = (train_x * 2 - 1).reshape(-1, 1, 28, 28)
    test_p = (test_x * 2 - 1).reshape(-1, 1, 28, 28)

    ckpt = torch.load(args.vae_checkpoint, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
        args.latent_channels = int(ckpt.get("latent_channels", args.latent_channels))
        args.vae_base = int(ckpt.get("base", args.vae_base))
    else:
        sd = ckpt  # raw state_dict (what train_vae.py saves)
        try:
            args.latent_channels = int(sd["mu.weight"].shape[0])
            args.vae_base = int(sd["enc.0.0.weight"].shape[0])
        except Exception:
            pass
    vae = VAE(latent_channels=args.latent_channels, base=args.vae_base).to(device)
    vae.load_state_dict(sd, strict=True)
    vae.eval()
    for prm in vae.parameters():
        prm.requires_grad_(False)
    lat_h = 28 // vae.downsample
    shape = (args.latent_channels, lat_h, lat_h)
    log(f"[vae] frozen {args.vae_checkpoint} -> latent {shape}")

    train_z = encode_data(vae, train_p, device)
    eval_z = encode_data(vae, test_p[:2000], device)
    log(f"[latent] train_z {tuple(train_z.shape)} mu_std {train_z.std().item():.3f}")
    ceil = F.mse_loss(decode_data(vae, eval_z[:64]), test_p[:64]).item()
    log(f"[vae] ceiling recon MSE[-1,1] = {ceil:.5f} (FM cannot beat this)")
    fm, ema, hist = run_loop(args, device, args.out_dir, vae,
                             train_z, eval_z, test_p, shape, ceil, t0)
    with open(os.path.join(args.out_dir, "fm_meta.json"), "w") as f:
        json.dump({"args": vars(args), "history": hist,
                   "elapsed_s": time.time() - t0}, f, indent=2)
    log(f"\nDONE in {time.time()-t0:.0f}s -> {args.out_dir}/ "
        f"(test {hist[-1]['test_mse']:.4f}, gen_px {hist[-1]['gen_px_mse']:.4f})")
    return hist[-1]


# Training loop
def run_loop(args, device, out, vae, train_z, eval_z, test_p, shape, ceil, t0):
    fm = FM_Conv(shape, base=args.base, depth=args.depth, tdim=args.tdim).to(device)
    npar = sum(p.numel() for p in fm.parameters())
    log(f"[fm] residual FiLM net: {npar/1e3:.0f}k params, base {args.base} depth {args.depth}")
    opt = torch.optim.AdamW(fm.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total = args.epochs * args.steps
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total, eta_min=args.lr_min)
    ema = EMA(fm, args.ema_decay)
    zb = torch.empty(args.batch, *shape, device=device)
    hist = []
    edges, tb0 = tbucket_report(fm, train_z, device)
    snap = sample_fm(ema.ema, 64, args.sample_steps, 64, device, shape, args.sampler)
    montage(decode_data(vae, snap), os.path.join(out, "samples_init.png"), title="init")
    for ep in range(args.epochs):
        fm.train()
        agg = 0.0
        for _ in range(args.steps):
            idx = torch.randint(0, len(train_z), (args.batch,))
            zb.copy_(train_z[idx], non_blocking=True)
            loss, _, _ = fm_step(fm, zb, device, args.t_sampling, args.sigma, args.w_clip)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(fm.parameters(), args.grad_clip)
            opt.step(); sched.step(); ema.update(fm)
            agg += loss.item()
        agg /= args.steps
        tr, te = eval_mse(fm, train_z, eval_z, device, t_mode=args.t_sampling)
        gen = sample_fm(ema.ema, 256, args.sample_steps, 64, device, shape, args.sampler)
        gidx = torch.randint(0, len(test_p), (256,))
        gpx = F.mse_loss(decode_data(vae, gen), test_p[gidx]).item()
        row = {"epoch": ep + 1, "train_mse": agg, "test_mse": te,
               "gen_px_mse": gpx, "vae_ceil_mse": ceil,
               "lr": sched.get_last_lr()[0], "elapsed_s": time.time() - t0}
        hist.append(row)
        log(f"[ep {ep+1:3d}/{args.epochs}] train {agg:.4f} test {te:.4f} "
            f"gen_px {gpx:.4f} (ceil {ceil:.4f}) lr {row['lr']:.1e} ({row['elapsed_s']:.0f}s)")
        if (ep + 1) % args.viz_every == 0 or ep == args.epochs - 1:
            snap = sample_fm(ema.ema, 64, args.sample_steps, 64, device, shape, args.sampler)
            montage(decode_data(vae, snap),
                    os.path.join(out, f"samples_ep{ep+1}.png"),
                    title=f"ep{ep+1} EMA {args.sampler}")
            plot_scores(hist, os.path.join(out, "loss_curve.png"))
    edges, tb1 = tbucket_report(ema.ema, train_z, device)
    plot_tbuckets(edges, [("init", tb0), ("trained", tb1)],
                  os.path.join(out, "tbucket_curve.png"))
    final = sample_fm(ema.ema, 256, args.sample_steps, 64, device, shape, args.sampler)
    montage(decode_data(vae, final), os.path.join(out, "samples_grid.png"), title="final")
    torch.save({"state_dict": fm.state_dict(), "shape": shape,
                "base": args.base, "depth": args.depth, "tdim": args.tdim},
               os.path.join(out, "fm.pt"))
    torch.save({"state_dict": ema.ema.state_dict(), "shape": shape,
                "base": args.base, "depth": args.depth, "tdim": args.tdim},
               os.path.join(out, "fm_ema.pt"))
    return fm, ema, hist


def main():
    run_training(build_parser().parse_args())


if __name__ == "__main__":
    main()










