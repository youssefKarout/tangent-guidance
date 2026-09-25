"""FID (Inception-V3 pool3, 2048-d) and its mean/trace decomposition.

Feature pipeline: grayscale image replicated to 3 channels, resized to
299x299, ImageNet normalization, pool3 activations captured with a forward
hook.
"""

import numpy as np
import torch
import torch.nn as nn

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class InceptionFeats(nn.Module):
    """FID feature extractor: torchvision inception_v3, 2048-d pool3 features."""

    def __init__(self, device):
        super().__init__()
        from torchvision.models import inception_v3, Inception_V3_Weights
        try:
            net = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1,
                               aux_logits=True, init_weights=False)
        except Exception:
            net = inception_v3(pretrained=True, aux_logits=True,
                               init_weights=False)
        net.eval()
        self.net = net.to(device)
        self.device = device
        self._pool_out = {}
        self.net.avgpool.register_forward_hook(self._grab_pool)

    def _grab_pool(self, module, inp, out):
        self._pool_out["f"] = torch.flatten(out, 1)

    @torch.no_grad()
    def feats(self, x01, batch=128):
        """x01: (N,1,28,28) in [0,1] -> (N,2048) pool3 features."""
        self.net.eval()
        out = []
        dev = self.device
        mean, std = IMAGENET_MEAN.to(dev), IMAGENET_STD.to(dev)
        for i in range(0, len(x01), batch):
            xb = x01[i:i + batch].to(dev)
            xb = xb.repeat(1, 3, 1, 1)
            if xb.shape[-1] != 299:
                xb = torch.nn.functional.interpolate(
                    xb, size=(299, 299), mode="bilinear", align_corners=False)
            self.net((xb - mean) / std)
            out.append(self._pool_out["f"].float().cpu())
        return torch.cat(out, dim=0)


def fid_from_feats(f1, f2):
    """FID(A, B) = ||mu1 - mu2||^2 + Tr(C1 + C2 - 2*sqrt(C1 C2))."""
    from scipy import linalg
    f1 = f1.numpy().astype(np.float64)
    f2 = f2.numpy().astype(np.float64)
    mu1, mu2 = f1.mean(0), f2.mean(0)
    s1 = np.cov(f1, rowvar=False)
    s2 = np.cov(f2, rowvar=False)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(s1.dot(s2), disp=False)
    if not np.isfinite(covmean).all():
        off = np.eye(s1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((s1 + off).dot(s2 + off), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(s1) + np.trace(s2)
                 - 2.0 * np.trace(covmean))


def fid_parts(f1, f2):
    """FID split into (total, mean_term, trace_term).

    mean_term  = ||mu1 - mu2||^2             shift of the feature centroid
    trace_term = Tr(C1 + C2 - 2 sqrt(C1C2))  difference in feature spread
    """
    from scipy import linalg
    f1 = f1.numpy().astype(np.float64)
    f2 = f2.numpy().astype(np.float64)
    mu1, mu2 = f1.mean(0), f2.mean(0)
    s1 = np.cov(f1, rowvar=False)
    s2 = np.cov(f2, rowvar=False)
    diff = mu1 - mu2
    mean_t = float(diff.dot(diff))
    covmean, _ = linalg.sqrtm(s1.dot(s2), disp=False)
    if not np.isfinite(covmean).all():
        off = np.eye(s1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((s1 + off).dot(s2 + off), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    trace_t = float(np.trace(s1) + np.trace(s2) - 2.0 * np.trace(covmean))
    return mean_t + trace_t, mean_t, trace_t
