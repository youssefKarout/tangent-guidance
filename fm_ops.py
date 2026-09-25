"""Data, latent encode/decode, flow-matching integration and kNN helpers.

These primitives contain no model definitions (see models.py). The sampling
pipeline uses them for the sequence noise -> FM rollout -> retrieval -> decode.
"""

import gzip
import os
import struct
import urllib.request

import numpy as np
import torch

MNIST_FILES = [("train-images-idx3-ubyte.gz", "train_x.npy"),
               ("train-labels-idx1-ubyte.gz", "train_y.npy"),
               ("t10k-images-idx3-ubyte.gz", "test_x.npy"),
               ("t10k-labels-idx1-ubyte.gz", "test_y.npy")]
MNIST_MIRRORS = ["https://ossci-datasets.s3.amazonaws.com/mnist/",
                 "https://storage.googleapis.com/cvdf-datasets/mnist/"]


def log(*a):
    print(*a, flush=True)


def _download(url, dest):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as f:
        f.write(resp.read())


def _parse_idx(gz_path, kind):
    with gzip.open(gz_path, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        n = struct.unpack(">I", f.read(4))[0]
        if kind == "images":
            if magic != 2051:
                raise ValueError(f"bad magic {magic} in {gz_path}")
            rows = struct.unpack(">I", f.read(4))[0]
            cols = struct.unpack(">I", f.read(4))[0]
            arr = np.frombuffer(f.read(), dtype=np.uint8)
            if arr.size != n * rows * cols:
                raise ValueError(f"truncated {gz_path}")
            return arr.reshape(n, rows * cols)
        if magic != 2049:
            raise ValueError(f"bad magic {magic} in {gz_path}")
        arr = np.frombuffer(f.read(), dtype=np.uint8)
        if arr.size != n:
            raise ValueError(f"truncated {gz_path}")
        return arr.astype(np.int64)


def load_mnist(data_dir, offline=False):
    """Return (train_x, train_y, test_x, test_y).

    Images have shape (N, 1, 28, 28) in [0, 1], whether they come from the
    .npy cache or from the downloaded idx.gz files.
    """
    raw_dir = os.path.join(data_dir, "mnist", "raw")
    npy_dir = os.path.join(data_dir, "mnist")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(npy_dir, exist_ok=True)
    paths = [os.path.join(npy_dir, out) for _, out in MNIST_FILES]

    def _imgs(a):
        return torch.from_numpy(np.asarray(a)).float().reshape(-1, 1, 28, 28)

    if all(os.path.exists(p) for p in paths):
        log(f"[data] cached MNIST from {npy_dir}")
        return (_imgs(np.load(paths[0])),
                torch.from_numpy(np.load(paths[1])).long(),
                _imgs(np.load(paths[2])),
                torch.from_numpy(np.load(paths[3])).long())
    if offline:
        raise RuntimeError("MNIST cache missing and --offline set")
    _EXP = {"train-images-idx3-ubyte.gz": 16 + 60000 * 784,
            "train-labels-idx1-ubyte.gz": 8 + 60000,
            "t10k-images-idx3-ubyte.gz": 16 + 10000 * 784,
            "t10k-labels-idx1-ubyte.gz": 8 + 10000}
    for fname, _ in MNIST_FILES:
        gz = os.path.join(raw_dir, fname)
        ok = False
        if os.path.exists(gz):
            try:
                with gzip.open(gz, "rb") as vf:
                    ok = len(vf.read()) == _EXP[fname]
            except Exception:
                ok = False
        if not ok:
            if os.path.exists(gz):
                os.remove(gz)
            last = None
            for base in MNIST_MIRRORS:
                try:
                    _download(base + fname, gz)
                    with gzip.open(gz, "rb") as vf:
                        if len(vf.read()) != _EXP[fname]:
                            raise ValueError("short download")
                    break
                except Exception as e:  # noqa: BLE001
                    last = e
                    if os.path.exists(gz):
                        os.remove(gz)
            else:
                raise RuntimeError(f"failed to download {fname}: {last}")
    tx = _parse_idx(os.path.join(raw_dir, MNIST_FILES[0][0]), "images") / 255.0
    ty = _parse_idx(os.path.join(raw_dir, MNIST_FILES[1][0]), "labels")
    ex = _parse_idx(os.path.join(raw_dir, MNIST_FILES[2][0]), "images") / 255.0
    ey = _parse_idx(os.path.join(raw_dir, MNIST_FILES[3][0]), "labels")
    for (_, out), arr in zip(MNIST_FILES, [tx, ty, ex, ey]):
        np.save(os.path.join(npy_dir, out), arr)
    return _imgs(tx), torch.from_numpy(ty).long(), _imgs(ex), torch.from_numpy(ey).long()


def make_synthetic_mnist(n, seed=0, hw=28):
    """Blobbed fake digits: for --use-synthetic code-path checks only."""
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



# Frozen-VAE encode/decode (deterministic: uses the posterior mean)
@torch.no_grad()
def encode_data(vae, x, device, batch=512):
    """pixel images in [-1,1] -> latent codes (posterior mean)."""
    vae.eval()
    outs = []
    for i in range(0, len(x), batch):
        outs.append(vae.encode(x[i:i + batch].to(device))[0].cpu())
    return torch.cat(outs, dim=0)


@torch.no_grad()
def decode_data(vae, z, batch=512):
    """latent codes -> pixel images in [-1,1] (on the VAE's device)."""
    vae.eval()
    dev = next(vae.parameters()).device
    outs = []
    z = z.to(dev)
    for i in range(0, len(z), batch):
        outs.append(vae.decode(z[i:i + batch]).cpu())
    return torch.cat(outs, dim=0)


# Flow-matching integration and vector helpers (t has shape (B, 1))
def normalize_dir(v, eps=1e-8):
    vf = v.flatten(1)
    return (vf / vf.norm(dim=-1, keepdim=True).clamp_min(eps)).reshape(v.shape)


def t_at(b, value, device):
    """(B,1) time tensor -- the shape every model forward expects."""
    return torch.full((b, 1), float(value), device=device)


def fm_integrate(x, t0, fm, steps=8):
    """Euler integration of dx/dt = fm(x,t) from t0 to 1 (t0: (B,1) or float)."""
    x = x.clone()
    dt = (1.0 - t0) / steps
    for i in range(steps):
        t = t0 + i * dt if torch.is_tensor(t0) else torch.full(
            (x.shape[0], 1), t0 + i * dt, device=x.device)
        dt_step = dt.view(-1, 1, 1, 1) if x.dim() == 4 else dt
        x = x + dt_step * fm(x, t)
    return x


@torch.no_grad()
def knn_barycenter_batched(query, bank, k, batch=256):
    """Mean of the k nearest bank entries (latent L2), per query sample."""
    bank = bank.to(query.device)
    qf = query.flatten(1)
    bf = bank.flatten(1)
    outs = []
    for i in range(0, qf.shape[0], batch):
        q = qf[i:i + batch]
        idx = torch.cdist(q, bf).topk(k=min(k, bf.shape[0]),
                                      largest=False).indices
        outs.append(bf[idx].mean(dim=1))
    return torch.cat(outs, dim=0).reshape(query.shape)


@torch.no_grad()
def knn_bary_dist(x, bank, k, knn_batch=512):
    """Per-sample squared distance from x to the bank kNN barycenter."""
    tgt = knn_barycenter_batched(x, bank, k=k, batch=knn_batch)
    return ((x - tgt) ** 2).flatten(1).sum(dim=-1)
