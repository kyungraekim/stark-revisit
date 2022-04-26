from typing import Optional

import tensorflow as tf

tk = tf.keras


class MultiheadAttention(tk.layers.Layer):
    in_proj_weight: Optional[tf.Variable]
    in_proj_bias: Optional[tf.Variable]
    out_proj_weight: Optional[tf.Variable]
    out_proj_bias: Optional[tf.Variable]

    def __init__(self, embed_dim, num_heads, dropout=0.0, **kwargs):
        super().__init__(**kwargs)

        self.model_dim = embed_dim
        self.num_heads = num_heads

        assert embed_dim % num_heads == 0
        self.head_dim = embed_dim // num_heads

        self.dropout = tk.layers.Dropout(rate=dropout)

    def build(self, input_shapes):
        in_dim = sum([shape[-1] for shape in input_shapes[:3]])

        self.in_proj_weight = self.add_weight(
            name='in_proj_weight', shape=(in_dim, self.model_dim),
            initializer=tf.keras.initializers.GlorotUniform(), dtype=tf.float32, trainable=True
        )
        self.in_proj_bias = self.add_weight(
            name='in_proj_bias', shape=(in_dim,),
            initializer=tf.keras.initializers.GlorotUniform(), dtype=tf.float32, trainable=True
        )
        self.out_proj_weight = self.add_weight(
            name='out_proj.weight', shape=(self.model_dim, self.model_dim),
            initializer=tf.keras.initializers.GlorotUniform(), dtype=tf.float32, trainable=True
        )
        self.out_proj_bias = self.add_weight(
            name='out_proj.bias', shape=(self.model_dim,),
            initializer=tf.keras.initializers.GlorotUniform(), dtype=tf.float32, trainable=True
        )

    def call(self, qkv_tuple, attn_mask=None, key_padding_mask=None,
             need_weights=True, training=False):
        query, key, value = qkv_tuple
        batch_size = tf.shape(query)[1]
        target_len = tf.shape(query)[0]
        source_len = tf.shape(key)[0]

        w_q = self.in_proj_weight[:self.model_dim, :]
        b = self.in_proj_bias[:self.model_dim]

        WQ = tf.matmul(query, w_q, transpose_b=True) + b

        w_k = self.in_proj_weight[self.model_dim:2 * self.model_dim, :]
        b = self.in_proj_bias[self.model_dim:2 * self.model_dim]
        WK = tf.matmul(key, w_k, transpose_b=True) + b

        w_v = self.in_proj_weight[2 * self.model_dim:, :]
        b = self.in_proj_bias[2 * self.model_dim:]
        WV = tf.matmul(value, w_v, transpose_b=True) + b

        WQ = tf.reshape(WQ, [target_len, batch_size * self.num_heads, self.head_dim])
        WQ = tf.transpose(WQ, [1, 0, 2])

        WK = tf.reshape(WK, [source_len, batch_size * self.num_heads, self.head_dim])
        WK = tf.transpose(WK, [1, 0, 2])

        WV = tf.reshape(WV, [source_len, batch_size * self.num_heads, self.head_dim])
        WV = tf.transpose(WV, [1, 0, 2])

        WQ = WQ / math.sqrt(WQ.shape[-1])
        attn_output_weights = tf.matmul(WQ, WK, transpose_b=True)

        if attn_mask is not None:
            attn_output_weights += attn_mask

        if key_padding_mask is not None:
            attn_output_weights = tf.reshape(attn_output_weights,
                                             [batch_size, self.num_heads, target_len, source_len])
            key_padding_mask = tf.expand_dims(key_padding_mask, 1)
            key_padding_mask = tf.expand_dims(key_padding_mask, 2)
            key_padding_mask = tf.tile(key_padding_mask, [1, self.num_heads, target_len, 1])
            attn_output_weights = tf.where(key_padding_mask,
                                           tf.zeros_like(attn_output_weights) + float('-inf'),
                                           attn_output_weights)
            attn_output_weights = tf.reshape(attn_output_weights,
                                             [batch_size * self.num_heads, target_len, source_len])

        attn_output_weights = tf.nn.softmax(attn_output_weights, axis=-1)
        attn_output_weights = self.dropout(attn_output_weights, training=training)

        attn_output = tf.matmul(attn_output_weights, WV)
        attn_output = tf.transpose(attn_output, [1, 0, 2])
        attn_output = tf.reshape(attn_output, [target_len, batch_size, self.model_dim])
        attn_output = tf.matmul(attn_output, self.out_proj_weight,
                                transpose_b=True) + self.out_proj_bias

        if need_weights:
            attn_output_weights = tf.reshape(attn_output_weights,
                                             [batch_size, self.num_heads, target_len, source_len])
            avg_weights = tf.reduce_mean(attn_output_weights, axis=1)
            return attn_output, avg_weights

        return attn_output
