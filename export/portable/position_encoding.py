"""
Various positional encodings for the transformer.
"""
import math

import torch
from torch import nn


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        assert mask is not None
        not_mask = ~mask  # (b,h,w)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)  # cumulative sum along axis 1 (h axis) --> (b, h, w)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)  # cumulative sum along axis 2 (w axis) --> (b, h, w)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale  # 2pi * (y / sigma(y))
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale  # 2pi * (x / sigma(x))

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)  # (0,1,2,...,d/2)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t  # (b,h,w,d/2)
        pos_y = y_embed[:, :, :, None] / dim_t  # (b,h,w,d/2)
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)  # (b,h,w,d/2)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)  # (b,h,w,d/2)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)  # (b,h,w,d)
        return pos


def build_position_encoding(cfg):
    N_steps = cfg.MODEL.HIDDEN_DIM // 2
    if cfg.MODEL.POSITION_EMBEDDING in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    else:
        raise ValueError(f"not supported {cfg.MODEL.POSITION_EMBEDDING}")

    return position_embedding
