import tensorflow as tf
import tensorflow.keras as tk

from export.convertable.box_head import build_box_head
from export.convertable.position_encoding import build_position_encoding
from export.convertable.resnet import resnet50, FrozenBatchNormalization, resnet101
from export.convertable.torch_to_tf import get_embedding_weights, get_conv_weights
from export.convertable.transformer import build_transformer


class ShapeFixedEmbedding(tk.layers.Embedding):
    def build(self, input_shape=None):
        if input_shape is None:
            input_shape = [None]
        super(ShapeFixedEmbedding, self).build((*input_shape, self.input_dim))


class StarkS:
    def __init__(self, backbone, position_encoder, transformer, box_head, query, bottleneck):
        self.backbone = backbone
        self.position_encoder = position_encoder
        self.bottleneck = bottleneck
        self.transformer = transformer
        self.query_embed = query
        self.box_head = box_head
        self.feat_sz_s = int(box_head.feat_sz)
        self.feat_len_s = int(box_head.feat_sz ** 2)

    def import_torch_model(self, model):
        self.import_torch_feature_extractor(model)
        self.import_torch_transformer(model)

    def import_torch_transformer(self, model):
        self.transformer.import_torch_model(model.transformer)
        self.box_head.import_torch_model(model.box_head)
        self.query_embed.set_weights(get_embedding_weights(model.query_embed))

    def import_torch_feature_extractor(self, model):
        self.backbone.import_torch_model(model.backbone[0].body)
        self.position_encoder.import_torch_model(model.backbone[1])
        self.bottleneck.set_weights(get_conv_weights(model.bottleneck))

    def forward_backbone(self, search_region, region_mask):
        feature_backbone = self.backbone(search_region)
        channeled_float_mask = tf.cast(region_mask[..., tf.newaxis], tf.float32)
        resized_mask = tf.compat.v1.image.resize_nearest_neighbor(channeled_float_mask,
                                                                  feature_backbone.shape[1:3],
                                                                  align_corners=False,
                                                                  half_pixel_centers=False)
        resized_mask = tf.cast(resized_mask, tf.bool)[..., 0]
        position_embedding = self.position_encoder(resized_mask)

        # (B, H, W, C) -> (B, HW, C) -> (HW, B, C)
        feature = self.bottleneck(feature_backbone)
        feature = _bhwc_to_hwbc(feature)
        resized_mask = tf.reshape(resized_mask, (resized_mask.shape[0], -1))
        position_embedding = _bhwc_to_hwbc(position_embedding)
        return feature, resized_mask, position_embedding

    def forward_transformer(self, feature, mask, position):
        # Forward the transformer encoder and decoder
        # (1, B, N, C), (HW1 + HW2, B, C)
        output_embed, enc_mem = self.transformer((feature, mask, self.query_embed.weights[0], position))
        # Forward the corner head
        enc_opt = tf.transpose(enc_mem[-self.feat_len_s:], (1, 0, 2))  # (HW1 + HW2, B, C) -> (B, HW2, C)
        dec_opt = tf.squeeze(output_embed, 0)  # (1, B, N, C) -> (B, N, C)
        att = tf.matmul(enc_opt, dec_opt, transpose_b=True)  # (B, HW2, N)
        opt = tf.expand_dims(enc_opt, -2) * tf.expand_dims(att, -1)  # (B, HW2, N, C)
        opt = tf.transpose(opt, (0, 2, 1, 3))  # (B, HW2, N, C) -> (B, N, HW2, C)
        b, n, hw, c = opt.shape
        opt = tf.reshape(opt, (-1, self.feat_sz_s, self.feat_sz_s, c))
        output_coord = self.box_head(opt)
        x0, y0, x1, y1 = tf.unstack(output_coord, axis=-1)
        cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
        return tf.stack([cx, cy, w, h], axis=-1)


def _bhwc_to_hwbc(tensor):
    """
    (B, H, W, C) -> (HW, B, C)
    """
    shape = tensor.shape
    tensor = tf.reshape(tensor, [shape[0], shape[1] * shape[2], shape[3]])
    return tf.transpose(tensor, (1, 0, 2))


def build_backbone(cfg):
    if cfg.MODEL.BACKBONE.TYPE == 'resnet50':
        backbone_creator = resnet50
    elif cfg.MODEL.BACKBONE.TYPE == 'resnet101':
        backbone_creator = resnet101
    else:
        raise AttributeError("Invalid TF-convertible backbone; {}".format(cfg.MODEL.BACKBONE.TYPE))

    return backbone_creator(**{
        'replace_stride_with_dilation': [False, cfg.MODEL.BACKBONE.DILATION, False],
        'norm_layer': FrozenBatchNormalization if cfg.TRAIN.FREEZE_BACKBONE_BN else None,
    })


def build_stark_s(cfg):
    position_embedding = build_position_encoding(cfg)

    backbone = build_backbone(cfg)
    transformer = build_transformer(cfg)
    box_head = build_box_head(cfg)

    query_embed = tk.layers.Embedding(cfg.MODEL.NUM_OBJECT_QUERIES, transformer.d_model)
    query_embed.build(None)
    bottleneck = tk.layers.Conv2D(transformer.d_model, 1)

    return StarkS(backbone, position_embedding, transformer, box_head, query_embed, bottleneck)
