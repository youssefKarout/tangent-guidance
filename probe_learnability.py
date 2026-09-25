"""Learnability probe for the teacher direction.

For one objective at a time, on the same states the FM trainer uses
(xt = (1 - t) * x0 + t * x1, with x1 drawn from the training bank), the probe
measures:

  mean_corr_norm   ||mean_i corr_i||        0 means the per-sample directions cancel
  shared_cos       cos(corr_i, mean corr)   ceiling of a constant predictor
  split_half_cos   cos(mean_A, mean_B)      whether the mean direction is stable
  ridge_cos        closed-form linear map x -> corr, on held-out states
  knn{k}_cos       mean of the K nearest training states' directions
  field_probe_*    a field trained on 3/4 of the states, evaluated on the other 1/4

latent_bary is the control: it is the objective a distilled field did learn. A
direction that is a function of x alone should score high on all of these;
pixel_tangent does not, which is why Stage B retrieves instead of distilling.
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import argparse
import time

import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fm_ops import encode_data, load_mnist, log
from models import _gn, load_fm, load_vae
from teacher import Teacher as TeacherOracle, build_pixel_bank


class FieldNet(nn.Module):
    """Time-independent residual conv field: x (B, C, H, W) -> (B, C, H, W)."""

    def __init__(self, shape, base=64, depth=3):
        super().__init__()
        channels = shape[0]
        self.inp = nn.Conv2d(channels, base, 3, padding=1)
        self.blocks = nn.ModuleList([
            nn.Sequential(_gn(base), nn.SiLU(), nn.Conv2d(base, base, 3, padding=1),
                          _gn(base), nn.SiLU(), nn.Conv2d(base, base, 3, padding=1))
            for _ in range(depth)])
        self.out = nn.Sequential(_gn(base), nn.SiLU(),
                                 nn.Conv2d(base, channels, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, x):
        h = self.inp(x)
        for block in self.blocks:
            h = h + block(h)
        return self.out(h)


def mean_cos(a, b, eps=1e-8):
    """cos between two (N, *shape) tensors, per sample, averaged."""
    af, bf = a.flatten(1), b.flatten(1)
    na = af.norm(dim=-1).clamp_min(eps)
    nb = bf.norm(dim=-1).clamp_min(eps)
    return float(((af * bf).sum(-1) / (na * nb)).mean())


def probe_states(bank, args, n, center, half_width, seed):
    """Same state distribution as the FM trainer: xt = (1 - t) * x0 + t * x1."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    x1 = bank[torch.randint(0, len(bank), (n,), generator=g)]
    x0 = torch.randn(n, *args._shape, generator=g)
    t = (center + (torch.rand(n, 1, generator=g) * 2 - 1) * half_width
         ).clamp(1e-3, 1 - 1e-3)
    tt = t.view(-1, 1, 1, 1)
    return ((1.0 - tt) * x0 + tt * x1).to(args.device), t.to(args.device)


@torch.no_grad()
def teacher_dirs(oracle, states, t, batch):
    """corr (unit teacher direction) for every state."""
    out = []
    for i in range(0, states.shape[0], batch):
        r = oracle.corr_at(states[i:i + batch], t[i:i + batch])
        out.append(r["corr"].detach().cpu())
    return torch.cat(out, dim=0)


def closed_form_probes(states, corr, frac=0.75, ks=(1, 5, 25), ridge=1e-2):
    """Optimization-free tests of whether corr is predictable from x.

    ridge_w : closed-form linear map x -> corr (no SGD, no initialization effects)
    knn_at_k: predict corr as the mean of the K nearest TRAIN states' directions;
              K=1 upper-bounds "local x implies the same direction"
    """
    n = states.shape[0]
    ntr = int(frac * n)
    X = states.flatten(1)
    C = corr.flatten(1)
    mu = X[:ntr].mean(0, keepdim=True)
    Xtr, Xte = X[:ntr] - mu, X[ntr:] - mu
    Ctr, Cte = C[:ntr], C[ntr:]
    lam = ridge * (Xtr.pow(2).sum(0).mean() / Xtr.shape[0]) * Xtr.shape[0]
    A = Xtr.t() @ Xtr + lam * torch.eye(Xtr.shape[1])
    W = torch.linalg.solve(A, Xtr.t() @ Ctr)
    rid = mean_cos(Xte @ W, Cte)
    out = {"ridge_cos": rid}
    d = torch.cdist(Xte, Xtr)
    order = d.argsort(dim=1)
    for k in ks:
        pred = Ctr[order[:, :k]].mean(dim=1)
        out[f"knn{k}_cos"] = mean_cos(pred, Cte)
    return out


def train_probe_field(states, corr, args, device, steps, seed=0):
    """Train a FieldNet on the same objective used for the distillation attempt."""
    torch.manual_seed(seed)
    n = states.shape[0]
    ntr = int(0.75 * n)
    xtr, ctr = states[:ntr], corr[:ntr]
    xte, cte = states[ntr:], corr[ntr:]
    field = FieldNet(args._shape, args.field_base, args.field_depth).to(device)
    opt = torch.optim.AdamW(field.parameters(), lr=args.field_lr,
                            weight_decay=args.weight_decay)
    hist = []
    for step in range(steps):
        i = torch.randint(0, ntr, (args.batch,), device=device)
        pred = field(xtr[i])
        pn = pred.flatten(1).norm(dim=-1, keepdim=True).clamp_min(1e-8)
        c = ctr[i]
        cos = 1 - (pred.flatten(1) * c.flatten(1)).sum(-1, keepdim=True) / pn
        mse = ((pred / pn.reshape(-1, 1, 1, 1) - c) ** 2
               ).mean(dim=[1, 2, 3], keepdim=True)
        loss = (args.cos_w * cos + args.mse_w * mse).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(field.parameters(), args.grad_clip)
        opt.step()
        if step % max(1, steps // 6) == 0 or step == steps - 1:
            hist.append((float(held_out_loss(field, xte, cte, args)),
                         float(held_out_cos(field, xte, cte))))
    return hist


@torch.no_grad()
def held_out_cos(field, x, c):
    p = field(x)
    pn = p.flatten(1).norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return (1 - (p.flatten(1) * c.flatten(1)).sum(-1, keepdim=True) / pn).mean()


@torch.no_grad()
def held_out_loss(field, x, c, args):
    p = field(x)
    pn = p.flatten(1).norm(dim=-1, keepdim=True).clamp_min(1e-8)
    cos = 1 - (p.flatten(1) * c.flatten(1)).sum(-1, keepdim=True) / pn
    mse = ((p / pn.reshape(-1, 1, 1, 1) - c) ** 2).mean(dim=[1, 2, 3],
                                                        keepdim=True)
    return (args.cos_w * cos + args.mse_w * mse).mean()



def main():
    p = argparse.ArgumentParser(description="teacher-direction learnability probe")
    p.add_argument("--vae-checkpoint", type=str, default=os.path.join("vae_run", "vae.pt"))
    p.add_argument("--fm-checkpoint", type=str, default=os.path.join("trained_fm", "fm_ema.pt"))
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--out-dir", type=str, default="probe_run")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--spaces", type=str,
                   default="latent_bary,pixel_tangent,pixel_tangent_k,pixel_bary")
    p.add_argument("--n", type=int, default=2048, help="states per space")
    p.add_argument("--bin", type=int, default=1, help="which t-bin (of --n-bins)")
    p.add_argument("--n-bins", type=int, default=3)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--probe-steps", type=int, default=300)
    p.add_argument("--field-base", type=int, default=64)
    p.add_argument("--field-depth", type=int, default=3)
    p.add_argument("--field-lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--cos-w", type=float, default=5.0)
    p.add_argument("--mse-w", type=float, default=1.5)
    p.add_argument("--teacher-steps", type=int, default=8)
    p.add_argument("--knn-k", type=int, default=25)
    p.add_argument("--prefilter", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache-dir", type=str, default="cache")
    args = p.parse_args()

    device = torch.device(args.device)
    args.device = device
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()
    train_x, _, _, _ = load_mnist(args.data_dir, args.offline)
    vae, shape = load_vae(args.vae_checkpoint, device)
    fm = load_fm(args.fm_checkpoint, device, shape)
    args._shape = shape
    lat = os.path.join(args.cache_dir, "bank_latents_train_60000.pt")
    bank = (torch.load(lat, map_location="cpu", weights_only=False)
            if os.path.exists(lat)
            else encode_data(vae, train_x * 2 - 1, device, batch=512))
    bank_pixels = build_pixel_bank(
        vae, bank, cache_path=os.path.join(args.cache_dir,
                                           f"bank_pixels_train_{len(bank)}.pt"))
    center = (args.bin + 0.5) / args.n_bins
    half = 0.5 / args.n_bins
    log(f"[setup] bank {tuple(bank.shape)} | bin {args.bin} center "
        f"{center:.3f} | n={args.n} | probe-steps {args.probe_steps}")

    results = {}
    for space in [s for s in args.spaces.split(",") if s]:
        oracle = TeacherOracle(fm, bank, space=space, vae=vae,
                               bank_pixels=bank_pixels, steps=args.teacher_steps,
                               k=args.knn_k, prefilter=args.prefilter)
        states, t = probe_states(bank, args, args.n, center, half, args.seed)
        corr = teacher_dirs(oracle, states, t, args.batch)
        mc = corr.mean(dim=0)
        hn = corr.shape[0] // 2
        res = {
            "mean_corr_norm": float(mc.flatten().norm()),
            "shared_cos": mean_cos(corr, mc.expand_as(corr)),
            "split_half_cos": mean_cos(corr[:hn].mean(0).unsqueeze(0),
                                       corr[hn:].mean(0).unsqueeze(0)),
        }
        res.update(closed_form_probes(states, corr))
        hist = train_probe_field(states, corr, args, device, args.probe_steps,
                                 seed=args.seed)
        res["field_probe_loss_curve"] = [h[0] for h in hist]
        res["field_probe_cos_curve"] = [h[1] for h in hist]
        results[space] = res
        log(f"[{space}] n={args.n} | ||mean corr||={res['mean_corr_norm']:.3f} "
            f"shared={res['shared_cos']:+.3f} split_half={res['split_half_cos']:+.3f} "
            f"| ridge={res['ridge_cos']:+.3f} knn1={res['knn1_cos']:+.3f} "
            f"knn5={res['knn5_cos']:+.3f} knn25={res['knn25_cos']:+.3f} "
            f"| field loss {[round(x, 3) for x in res['field_probe_loss_curve']]}"
            f" cos {[round(c, 3) for c in res['field_probe_cos_curve']]}"
            f" ({time.time() - t0:.0f}s)")

    import json
    with open(os.path.join(args.out_dir, "probe_summary.json"), "w") as f:
        json.dump({"args": {k: str(v) for k, v in vars(args).items()},
                   "results": results}, f, indent=2)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for k, v in results.items():
        ax[0].plot(v["field_probe_loss_curve"], marker="o", label=k)
        ax[1].plot(v["field_probe_cos_curve"], marker="o", label=k)
    for a_, ttl, yl in ((ax[0], "held-out loss (cos_w=5)", "loss"),
                        (ax[1], "held-out 1-cos(field, teacher)", "1-cos")):
        a_.set_title(ttl)
        a_.set_xlabel("probe checkpoint")
        a_.set_ylabel(yl)
        a_.grid(alpha=0.3)
        a_.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "probe_curve.png"), dpi=120)
    plt.close()
    log(f"DONE in {time.time() - t0:.0f}s -> {args.out_dir}/probe_summary.json")


if __name__ == "__main__":
    main()

