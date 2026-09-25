"""Retrieval teacher (Stage B): a non-parametric correction at the FM endpoint.

At a (point, time) pair the teacher looks up the nearest training latents in
the frozen-VAE bank and, for the pixel objectives, in the decoded-image bank.
Every objective returns a unit latent direction, so x <- x + eta * dir has the
same step size across variants.

  latent_bary     pull toward the kNN barycenter in latent L2
  pixel_bary      pull toward the barycenter of the decoded neighbours
  pixel_nn1       pull toward the single nearest decoded neighbour
  pixel_tangent   as pixel_nn1, with the radial (mean-seeking) part removed
  pixel_tangent_k as pixel_tangent, averaged over the k neighbours

The pixel objectives differentiate through the frozen decoder to convert a
pixel-space target into a latent direction.

The resulting direction depends on the retrieval bank and not on x alone;
probe_learnability.py measures that, which is why Stage B is a retrieval step
rather than a distilled parametric field. README.md lists the FID comparison
of the five objectives.
"""

import torch

from fm_ops import (decode_data, fm_integrate, knn_barycenter_batched,
                    normalize_dir)

SPACES = ("latent_bary", "pixel_bary", "pixel_nn1", "pixel_tangent",
          "pixel_tangent_k")


def is_pixel_space(space):
    return str(space).startswith("pixel")


@torch.no_grad()
def build_pixel_bank(vae, bank_latents, batch=512, cache_path=None):
    """Decode the real-latent bank once -> (N, 784) images in [-1, 1]."""
    import os
    if cache_path and os.path.exists(cache_path):
        return torch.load(cache_path, map_location="cpu", weights_only=False)
    outs = []
    for i in range(0, bank_latents.shape[0], batch):
        outs.append(decode_data(vae, bank_latents[i:i + batch]).flatten(1).cpu())
    pix = torch.cat(outs, dim=0)
    if cache_path:
        d = os.path.dirname(cache_path)
        if d:
            os.makedirs(d, exist_ok=True)
        torch.save(pix, cache_path)
    return pix


class Teacher:
    """Retrieval correction at a (point, time) pair.

    fm          frozen velocity field, used for the internal rollout
    bank        frozen-VAE latents of the training split
    space       one of SPACES
    vae         decoder, differentiated through for the pixel spaces
    bank_pixels decoded bank, see build_pixel_bank
    steps       rollout steps inside the probe
    k           number of neighbours
    prefilter   latent candidates kept before the pixel-space kNN (0 = exact)
    """

    def __init__(self, fm, bank, space="pixel_tangent", vae=None,
                 bank_pixels=None, steps=8, k=25, knn_batch=512,
                 prefilter=256, pixel_batch=64):
        self.fm = fm
        self.bank = bank
        self.space = str(space)
        if self.space not in SPACES:
            raise ValueError(f"unknown teacher space {space!r}; use one of {SPACES}")
        self.pixel = is_pixel_space(self.space)
        if self.pixel and (vae is None or bank_pixels is None):
            raise ValueError(f"{self.space} needs the VAE and a decoded pixel bank")
        self.vae = vae
        self.bank_pixels = bank_pixels
        self.steps = int(steps)
        self.k = int(k)
        self.knn_batch = int(knn_batch)
        self.prefilter = int(prefilter)
        self.pixel_batch = int(pixel_batch)

    def meta(self):
        return {"space": self.space, "steps": self.steps, "k": self.k,
                "prefilter": self.prefilter,
                "bank": int(self.bank.shape[0]),
                "bank_pixels": (None if self.bank_pixels is None
                                else int(self.bank_pixels.shape[0]))}

    @torch.no_grad()
    def _knn_pixel(self, x_lat, x_pix):
        """Neighbour indices in decoded-pixel space, prefiltered in latent space.

        The prefilter keeps this affordable: a full 60k-bank pixel cdist costs
        ~1e10 flops per batch, the prefiltered one ~5e7.
        """
        ks = 1 if self.space == "pixel_nn1" else self.k
        bl = self.bank.to(x_lat.device).flatten(1)
        bp = self.bank_pixels
        out = []
        for i in range(0, x_lat.shape[0], self.pixel_batch):
            ql = x_lat[i:i + self.pixel_batch].flatten(1)
            qp = x_pix[i:i + self.pixel_batch]
            if self.prefilter and self.prefilter < bl.shape[0]:
                cand = torch.cdist(ql, bl).topk(self.prefilter,
                                                largest=False).indices
                nb = bp.to(qp.device)[cand]                       # (b, M, 784)
                d = torch.cdist(qp.unsqueeze(1), nb).squeeze(1)   # (b, M)
                out.append(cand.gather(1, d.topk(min(ks, cand.shape[1]),
                                                 largest=False).indices))
            else:
                d = torch.cdist(qp, bp.to(qp.device))
                out.append(d.topk(min(ks, bp.shape[0]), largest=False).indices)
        return torch.cat(out, dim=0)

    def corr_at(self, x, t, target_only=False):
        """Correction at (x, t).

        Returns a dict:
          corr  (B, *shape) unit latent direction added by the sampler
          grad  (B, *shape) -grad of the teacher loss; its norm is the step scale
          base  (B,)        per-sample loss in the teacher's own space
          diag  extra scalars, e.g. tangent_keep = fraction of the pull retained
        """
        with torch.enable_grad():
            xt = x.detach().clone().requires_grad_(True)
            xhat = fm_integrate(xt, t, self.fm, self.steps)
            diag = {}
            if not self.pixel:
                target = knn_barycenter_batched(xhat.detach(), self.bank,
                                                k=self.k, batch=self.knn_batch)
                base = ((xhat - target) ** 2).flatten(1).sum(dim=-1)
            else:
                pixf = self.vae.decode(xhat).flatten(1)            # differentiable
                pix_d = pixf.detach()
                idx = self._knn_pixel(xhat.detach(), pix_d)
                nbr = self.bank_pixels.to(pixf.device)[idx]
                if self.space == "pixel_bary":
                    target = nbr.mean(dim=1)
                elif self.space == "pixel_nn1":
                    target = nbr[:, 0]
                elif self.space == "pixel_tangent_k":
                    m = nbr.mean(dim=1)
                    u = pix_d - m
                    u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                    d = nbr - pixf.unsqueeze(1)
                    d = d - (d * u.unsqueeze(1)).sum(-1, keepdim=True) * u.unsqueeze(1)
                    dm = d.mean(dim=1)
                    target = pix_d + dm
                    d0 = (nbr.mean(dim=1) - pixf).norm(dim=-1).clamp_min(1e-8)
                    diag["tangent_keep"] = (dm.norm(dim=-1) / d0).mean().item()
                else:                                              # pixel_tangent
                    m = nbr.mean(dim=1) if nbr.shape[1] > 1 else nbr[:, 0]
                    u = pix_d - m
                    u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                    d = nbr[:, 0] - pixf
                    d = d - (d * u).sum(-1, keepdim=True) * u
                    target = pix_d + d
                    d0 = (nbr[:, 0] - pixf).norm(dim=-1).clamp_min(1e-8)
                    diag["tangent_keep"] = (d.norm(dim=-1) / d0).mean().item()
                base = ((pixf - target) ** 2).flatten(1).sum(dim=-1)
            if target_only:
                return {"target": target.detach(), "base": base.detach(),
                        "diag": diag}
            loss = base.mean()
            grad = torch.autograd.grad(loss, xt, retain_graph=False,
                                       create_graph=False)[0]
        return {"corr": normalize_dir(-grad).detach(), "grad": grad.detach(),
                "base": base.detach(), "target": target.detach(), "diag": diag}

