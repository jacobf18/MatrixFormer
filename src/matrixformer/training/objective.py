"""Full-support density, mean Huber, and discrete CDF losses."""

from importlib.resources import files
import math

import torch
from torch import nn
from torch.nn import functional


class DistributionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        with files("matrixformer").joinpath("assets/borders.pt").open("rb") as handle:
            borders = torch.load(handle, map_location="cpu", weights_only=True).float()
        widths = borders[1:] - borders[:-1]
        midpoints = borders[:-1] + widths / 2
        means = midpoints.clone()
        tail_scales = widths[[0, -1]] / 0.6744897501960817
        means[0] = borders[1] - tail_scales[0] * math.sqrt(2 / math.pi)
        means[-1] = borders[-2] + tail_scales[1] * math.sqrt(2 / math.pi)
        for name, value in dict(borders=borders, log_widths=widths.log(), midpoints=midpoints,
                                means=means, tail_scales=tail_scales).items():
            self.register_buffer(name, value)

    def mean(self, logits):
        return logits.float().softmax(-1) @ self.means

    def nll(self, logits, truth):
        indices = (torch.searchsorted(self.borders, truth.contiguous()) - 1).clamp(0, self.means.numel() - 1)
        categorical = functional.cross_entropy(logits.float(), indices, reduction="none")
        density = self.log_widths[indices]
        left = (self.borders[1] - truth).clamp_min(1e-8)
        right = (truth - self.borders[-2]).clamp_min(1e-8)
        tail_constant = 0.5 * math.log(math.pi / 2)
        left_nll = tail_constant + self.tail_scales[0].log() + 0.5 * (left / self.tail_scales[0]).square()
        right_nll = tail_constant + self.tail_scales[1].log() + 0.5 * (right / self.tail_scales[1]).square()
        density = torch.where(indices == 0, left_nll, density)
        density = torch.where(indices == self.means.numel() - 1, right_nll, density)
        return categorical + density

    def forward(self, logits, truth, *, domain):
        logits, truth = logits.float(), truth.float()
        if logits.ndim != 2 or logits.shape != (truth.numel(), self.means.numel()):
            raise ValueError("Expected one distribution per selected cell")
        probabilities = logits.softmax(-1)
        mean = probabilities @ self.means
        huber = functional.huber_loss(mean, truth, delta=1.0, reduction="none")
        cdf_target = (self.midpoints[None] >= truth[:, None]).float()
        cdf = (probabilities.cumsum(-1) - cdf_target).square().mean(-1)
        truth_weight = {"science": 4.0, "imputation": 2.0, "recsys": 1.0, "score_panel": 1.0}[domain]
        nll_weight = 0.2 if domain == "recsys" else 1.0
        return nll_weight * self.nll(logits, truth) + truth_weight * huber + 0.05 * cdf
