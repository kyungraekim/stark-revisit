import math

import tensorflow as tf
import tensorflow.keras as tk


class PositionEmbeddingSine(tk.Model):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=1000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def call(self, pos_mask, training=False, mask=None):
        not_mask = tf.cast(~pos_mask, tf.float32)  # (b,h,w)
        y_embed = tf.math.cumsum(not_mask, axis=1)  # cumulative sum along axis 1 (h axis) --> (b, h, w)
        x_embed = tf.math.cumsum(not_mask, axis=2)  # cumulative sum along axis 2 (w axis) --> (b, h, w)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale  # 2pi * (y / sigma(y))
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale  # 2pi * (x / sigma(x))

        dim_t = tf.range(self.num_pos_feats, dtype=tf.float32)  # (0,1,2,...,d/2)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, tf.newaxis] / dim_t  # (b,h,w,d/2)
        pos_y = y_embed[:, :, :, tf.newaxis] / dim_t  # (b,h,w,d/2)
        pos_x = tf.stack((tf.math.sin(pos_x[:, :, :, 0::2]), tf.math.cos(pos_x[:, :, :, 1::2])), axis=4)  # (b,h,w,d/2)
        pos_y = tf.stack((tf.math.sin(pos_y[:, :, :, 0::2]), tf.math.cos(pos_y[:, :, :, 1::2])), axis=4)  # (b,h,w,d/2)
        shape = [*pos_x.shape[:3], -1]
        pos_x = tf.reshape(pos_x, shape)
        pos_y = tf.reshape(pos_y, shape)
        pos = tf.concat((pos_y, pos_x), axis=3)  # (b,h,w,d)
        return pos

    def import_torch_model(self, model):
        self.num_pos_feats = model.num_pos_feats
        self.temperature = model.temperature
        self.normalize = model.normalize
        self.scale = model.scale


def build_position_encoding(cfg):
    return build_position_encoding_with_dim(cfg.MODEL.HIDDEN_DIM // 2)


def build_position_encoding_with_dim(dim):
    return PositionEmbeddingSine(dim, normalize=True)
