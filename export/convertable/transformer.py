from typing import Optional

import tensorflow.keras as tk
from tensorflow import Tensor

from export.convertable.attention import MultiheadAttention


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return tk.activations.relu
    if activation == "gelu":
        return tk.activations.gelu
    if activation == "glu":
        return tk.activations.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def _get_clones(model, n):
    return [tk.models.clone_model(model) for _ in range(n)]


class TransformerEncoder(tk.layers.Layer):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def call(self, src,
             mask: Optional[Tensor] = None,
             src_key_padding_mask: Optional[Tensor] = None,
             pos: Optional[Tensor] = None,
             return_intermediate=False):
        if return_intermediate:
            output_list = []
            output = src

            for layer in self.layers:
                output = layer(output, src_mask=mask,
                               src_key_padding_mask=src_key_padding_mask, pos=pos)
                if self.norm is None:
                    output_list.append(output)
            if self.norm is not None:
                output = self.norm(output)
                output_list.append(output)
            return output_list
        else:
            output = src

            for layer in self.layers:
                output = layer(output, src_mask=mask,
                               src_key_padding_mask=src_key_padding_mask, pos=pos)

            if self.norm is not None:
                output = self.norm(output)

            return output


class TransformerEncoderLayer(tk.layers.Layer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, divide_norm=False):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = tk.layers.Linear(dim_feedforward)
        self.dropout = tk.layers.Dropout(dropout)
        self.linear2 = tk.layers.Linear(d_model)

        self.norm1 = tk.layers.LayerNormalization(d_model)
        self.norm2 = tk.layers.LayerNormalization(d_model)
        self.dropout1 = tk.layers.Dropout(dropout)
        self.dropout2 = tk.layers.Dropout(dropout)

        self.activation = tk.layers.Activation(activation)
        if normalize_before:
            raise AttributeError("Unsupported option; normalize_before == True")
        if divide_norm:
            raise AttributeError("Unsupported option; divide_norm == True")
        self.scale_factor = float(d_model // nhead) ** 0.5

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)  # add pos to src
        src2 = self.self_attn(q, k, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def call(self, src,
             src_mask: Optional[Tensor] = None,
             src_key_padding_mask: Optional[Tensor] = None,
             pos: Optional[Tensor] = None):
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)
