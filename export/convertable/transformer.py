from typing import Optional

import tensorflow as tf
import tensorflow.keras as tk

from export.convertable import attention
from export.convertable.attention import MultiheadAttention


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return tk.activations.relu
    raise RuntimeError(f"activation should be relu, not {activation}.")


class Transformer(tk.Model):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, divide_norm=False):
        super().__init__()
        self.encoder = TransformerEncoder(d_model, nhead, dim_feedforward, dropout, activation,
                                          num_encoder_layers)
        self.decoder = TransformerDecoder(d_model, nhead, dim_feedforward, dropout, activation,
                                          num_decoder_layers)

        self.d_model = d_model
        self.nhead = nhead
        self.d_feed = dim_feedforward
        self.scale_factor = float(d_model // nhead) ** 0.5

    def call(self, feat, mask, query_embed, pos_embed):
        """

        :param feat: (H1W1+H2W2, bs, C)
        :param mask: (bs, H1W1+H2W2)
        :param query_embed: (N, C) or (N, B, C)
        :param pos_embed: (H1W1+H2W2, bs, C)
        :return:
        """
        memory = self.encoder(feat, src_key_padding_mask=mask, pos=pos_embed)
        assert len(query_embed.shape) in [2, 3]
        if len(query_embed.shape) == 2:
            batch_size = feat.shape[1]
            # (N,C) --> (N,1,C) --> (N,B,C)
            query_embed = tf.tile(tf.expand_dims(query_embed, 1), (1, batch_size, 1))
        tgt = tf.zeros_like(query_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        return hs, memory

    def import_torch_model(self, model):
        self.encoder.import_torch_model(model.encoder)
        self.decoder.import_torch_model(model.decoder)


class TransformerEncoder(tk.Model):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation, num_layers):
        super(TransformerEncoder, self).__init__()
        self.enc_layers = [
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
            for _ in range(num_layers)
        ]
        self.num_layers = num_layers
        self.norm = tk.layers.LayerNormalization(epsilon=1e-5)

    def call(self, src,
             mask: Optional[tf.Tensor] = None,
             src_key_padding_mask: Optional[tf.Tensor] = None,
             pos: Optional[tf.Tensor] = None):
        output = src

        for layer in self.enc_layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output

    def import_torch_model(self, model):
        for layer, torch_layer in zip(self.enc_layers, model.layers):
            layer.import_torch_model(torch_layer)


class TransformerEncoderLayer(tk.layers.Layer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.dropout1 = tk.layers.Dropout(dropout)
        self.dropout2 = tk.layers.Dropout(dropout)
        self.norm1 = tk.layers.LayerNormalization(epsilon=1e-5)
        self.norm2 = tk.layers.LayerNormalization(epsilon=1e-5)
        self.linear1 = tk.layers.Dense(dim_feedforward)
        self.linear2 = tk.layers.Dense(d_model)
        self.dropout = tk.layers.Dropout(dropout)
        self.activation = tk.layers.Activation(activation)
        self.scale_factor = float(d_model // nhead) ** 0.5

    def with_pos_embed(self, tensor, pos: Optional[tf.Tensor]):
        return tensor if pos is None else tensor + pos

    def call(self,
             src,
             src_mask: Optional[tf.Tensor] = None,
             src_key_padding_mask: Optional[tf.Tensor] = None,
             pos: Optional[tf.Tensor] = None):
        q = k = self.with_pos_embed(src, pos)  # add pos to src
        src2 = self.self_attn((q, k, src), attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def import_torch_model(self, model):
        module = 'self_attn'
        tf_model = getattr(self, module)
        torch_module = getattr(model, module)
        tf_model.import_torch_model(torch_module)

        for layer in ['linear1', 'linear2']:
            tf_layer = getattr(self, layer)
            torch_layer = getattr(model, layer)
            tf_layer.set_weights(get_linear_weights(torch_layer))


class TransformerDecoder(tk.Model):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation, num_layers):
        super(TransformerDecoder, self).__init__()
        self.dec_layers = [
            TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
            for _ in range(num_layers)
        ]
        self.num_layers = num_layers
        self.norm = tk.layers.LayerNormalization(epsilon=1e-5)

    def call(self, tgt, memory,
             tgt_mask: Optional[tf.Tensor] = None,
             memory_mask: Optional[tf.Tensor] = None,
             tgt_key_padding_mask: Optional[tf.Tensor] = None,
             memory_key_padding_mask: Optional[tf.Tensor] = None,
             pos: Optional[tf.Tensor] = None,
             query_pos: Optional[tf.Tensor] = None):
        output = tgt

        for layer in self.dec_layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)

        if self.norm is not None:
            output = self.norm(output)

        return tf.expand_dims(output, 0)

    def import_torch_model(self, model):
        for layer, torch_layer in zip(self.dec_layers, model.layers):
            layer.import_torch_model(torch_layer)


class TransformerDecoderLayer(tk.layers.Layer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        super().__init__()
        self.self_attn = attention.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = attention.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = tk.layers.Dense(dim_feedforward)
        self.dropout = tk.layers.Dropout(dropout)
        self.linear2 = tk.layers.Dense(d_model)

        self.norm1 = tk.layers.LayerNormalization(epsilon=1e-5)
        self.norm2 = tk.layers.LayerNormalization(epsilon=1e-5)
        self.norm3 = tk.layers.LayerNormalization(epsilon=1e-5)
        self.dropout1 = tk.layers.Dropout(dropout)
        self.dropout2 = tk.layers.Dropout(dropout)
        self.dropout3 = tk.layers.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        self.scale_factor = float(d_model // nhead) ** 0.5

    def with_pos_embed(self, tensor, pos: Optional[tf.Tensor]):
        return tensor if pos is None else tensor + pos

    def call(self, tgt, memory,
             tgt_mask: Optional[tf.Tensor] = None,
             memory_mask: Optional[tf.Tensor] = None,
             tgt_key_padding_mask: Optional[tf.Tensor] = None,
             memory_key_padding_mask: Optional[tf.Tensor] = None,
             pos: Optional[tf.Tensor] = None,
             query_pos: Optional[tf.Tensor] = None):
        # self-attention
        q = k = self.with_pos_embed(tgt, query_pos)  # Add object query to the query and key
        tgt2 = self.self_attn((q, k, tgt), attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # mutual attention
        queries, keys = self.with_pos_embed(tgt, query_pos), self.with_pos_embed(memory, pos)
        tgt2 = self.multihead_attn((queries, keys, memory), attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def import_torch_model(self, model):
        for module in ['self_attn', 'multihead_attn']:
            tf_model = getattr(self, module)
            torch_module = getattr(model, module)
            tf_model.import_torch_model(torch_module)

        for layer in ['linear1', 'linear2']:
            tf_layer = getattr(self, layer)
            torch_layer = getattr(model, layer)
            tf_layer.set_weights(get_linear_weights(torch_layer))


def get_weights(layer, names):
    return [getattr(layer, name).detach().numpy() for name in names]


def get_linear_weights(layer):
    weights = get_weights(layer, ['weight', 'bias'])
    weights[0] = weights[0].transpose()
    return weights


def build_transformer(cfg):
    return Transformer(
        d_model=cfg.MODEL.HIDDEN_DIM,
        dropout=cfg.MODEL.TRANSFORMER.DROPOUT,
        nhead=cfg.MODEL.TRANSFORMER.NHEADS,
        dim_feedforward=cfg.MODEL.TRANSFORMER.DIM_FEEDFORWARD,
        num_encoder_layers=cfg.MODEL.TRANSFORMER.ENC_LAYERS,
        num_decoder_layers=cfg.MODEL.TRANSFORMER.DEC_LAYERS,
        normalize_before=cfg.MODEL.TRANSFORMER.PRE_NORM,
        return_intermediate_dec=False,  # we use false to avoid DDP error,
        divide_norm=cfg.MODEL.TRANSFORMER.DIVIDE_NORM
    )
