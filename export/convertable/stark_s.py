import tensorflow as tf
import tensorflow.keras as tk

from export.convertable.box_head import build_box_head
from export.convertable.position_encoding import build_position_encoding
from export.convertable.resnet import resnet50, FrozenBatchNormalization, resnet101
from export.convertable.torch_to_tf import get_embedding_weights, get_conv_weights
from export.convertable.transformer import build_transformer


class StarkS:
    def __init__(self, backbone, position_encoder, transformer, box_head, query, bottleneck):
        self.backbone = backbone
        self.position_encoder = position_encoder
        self.bottleneck = bottleneck
        self.transformer = transformer
        self.query_embed = query
        self.box_head = box_head

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
    bottleneck = tk.layers.Conv2D(transformer.d_model, 1)

    return StarkS(backbone, position_embedding, transformer, box_head, query_embed, bottleneck)
