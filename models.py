"""Model definitions and checkpoint loaders.

    VAE     : x (1x28x28, [-1,1]) <-> z (C x 7 x 7)   -- train_vae.py
    FM_Conv : velocity field on the latent map        -- train_fm.py

`load_vae` / `load_fm` rebuild the architecture from checkpoint metadata and
return an eval-mode module with gradients disabled, so every downstream script
can do

    vae, shape = load_vae("vae_run/vae.pt", device)
    fm         = load_fm("trained_fm/fm_ema.pt", device, shape)
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOGVAR_MIN, LOGVAR_MAX = -6.0, 4.0


def _gn(ch, max_groups=8):
    """GroupNorm with a group count that divides `ch` (robust to small bases)."""
    g = min(max_groups, ch)
    while ch % g:
        g -= 1
    return nn.GroupNorm(g, ch)


# VAE (same architecture as train_vae.py)
class VAE(nn.Module):
    """Conv VAE: x (1x28x28, [-1,1]) <-> z (C x 7 x 7).

    Encoder  : two stride-2 convs (28 -> 14 -> 7) + GroupNorm/SiLU, 1x1-conv heads
    Decoder  : conv + two stride-2 transposed convs (7 -> 14 -> 28), tanh output
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


# Flow-matching velocity field: sinusoidal time embedding + FiLM residual blocks
class TimeEmbed(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(),
                                 nn.Linear(dim * 4, dim * 4))

    def forward(self, t):
        t = t.reshape(-1)
        half = self.dim // 2
        freqs = torch.exp(-np.log(10000.0)
                          * torch.arange(half, device=t.device) / half)
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
        nn.init.zeros_(self.c2.weight)
        nn.init.zeros_(self.c2.bias)

    def forward(self, h, te):
        s, b = self.film(te).chunk(2, dim=1)
        s = s[:, :, None, None]
        b = b[:, :, None, None]
        h0 = h
        h = self.c1(F.silu(self.n1(h)))
        h = self.n2(h) * (1 + s) + b
        h = self.c2(F.silu(h))
        return h0 + h


class FM_Conv(nn.Module):
    """Velocity field on the latent map: (B,C,H,W), t -> (B,C,H,W)."""

    def __init__(self, latent_shape=(8, 7, 7), base=64, depth=4, tdim=128):
        super().__init__()
        c = latent_shape[0]
        self.inp = nn.Conv2d(c, base, 3, padding=1)
        self.temb = TimeEmbed(tdim)
        self.blocks = nn.ModuleList([ResBlock(base, tdim * 4)
                                     for _ in range(depth)])
        self.out = nn.Sequential(_gn(base), nn.SiLU(),
                                 nn.Conv2d(base, c, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, x, t):
        te = self.temb(t.reshape(-1))
        h = self.inp(x)
        for blk in self.blocks:
            h = blk(h, te)
        return self.out(h)


# Checkpoint loaders (architecture is read from the checkpoint)
def load_vae(path, device, downsample=None):
    """Returns (vae, (C, H, W)). Accepts {'state_dict': ...} or a raw state_dict."""
    ck = torch.load(path, map_location=device, weights_only=False)
    meta = {}
    if isinstance(ck, dict) and "state_dict" in ck:
        meta = {k: v for k, v in ck.items() if k != "state_dict"}
        ck = ck["state_dict"]
    lc = int(meta.get("latent_channels", ck["mu.weight"].shape[0]))
    vb = int(meta.get("base", ck["enc.0.0.weight"].shape[0]))
    ds = int(downsample or meta.get("downsample", 4))
    vae = VAE(1, lc, ds, vb).to(device)
    vae.load_state_dict(ck, strict=True)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae, (lc, 28 // ds, 28 // ds)


def load_fm(path, device, shape=None):
    """Return the FM in eval mode; architecture is read from the checkpoint."""
    ck = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ck, dict) and "state_dict" in ck:
        meta = {k: v for k, v in ck.items() if k != "state_dict"}
        sd = ck["state_dict"]
    else:
        meta, sd = {}, ck
    shp = tuple(meta.get("shape", shape or (sd["inp.weight"].shape[0], 7, 7)))
    base = int(meta.get("base", sd["inp.weight"].shape[0]))
    tdim = int(meta.get("tdim", sd["temb.mlp.0.weight"].shape[0]))
    depth = int(meta.get("depth",
                         len([k for k in sd if k.endswith(".film.weight")])))
    fm = FM_Conv(shp, base, depth, tdim).to(device)
    fm.load_state_dict(sd, strict=True)
    fm.eval()
    for p in fm.parameters():
        p.requires_grad_(False)
    return fm

