"""
Basic STARK Model (Spatial-only).
"""
import torch
import torch.nn as nn

from lib.utils.box_ops import box_xyxy_to_cxcywh
from .backbone import build_backbone
from .box_head import build_box_head
from .error import ExportError
from .transformer import build_transformer


class STARKS(nn.Module):
    """ This is the base class for Transformer Tracking """

    def __init__(self, backbone, transformer, box_head, num_queries,
                 aux_loss=False, head_type="CORNER"):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_queries: number of object queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.box_head = box_head
        self.num_queries = num_queries
        hidden_dim = transformer.d_model
        self.query_embed = nn.Embedding(num_queries, hidden_dim)  # object queries
        self.bottleneck = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)  # the bottleneck layer
        if aux_loss:
            raise ExportError('aux_loss')
        self.head_type = head_type
        if head_type != "CORNER":
            raise ExportError(head_type)
        self.feat_sz_s = int(box_head.feat_sz)
        self.feat_len_s = int(box_head.feat_sz ** 2)

    def forward_old(self, img=None, feat=None, mask=None, pos=None, mode="backbone"):
        if mode == "backbone":
            return self.forward_backbone(img, mask)
        elif mode == "transformer":
            return self.forward_transformer(feat, mask, pos)
        else:
            raise ValueError

    def forward(self, img: torch.Tensor, mask: torch.Tensor, template):
        feat, mask, pos = self.forward_backbone(img, mask)
        feat_t, mask_t, pos_t = template
        feat = torch.cat([feat_t, feat], dim=0)
        mask = torch.cat([mask_t, mask], dim=1)
        pos = torch.cat([pos_t, pos], dim=0)
        return self.forward_transformer(feat, mask, pos)

    def forward_backbone(self, input: torch.Tensor, mask: torch.Tensor):
        """The input type is NestedTensor, which consists of:
               - tensor: batched images, of shape [batch_size x 3 x H x W]
               - mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        """
        # Forward the backbone
        features, masks, pos = self.backbone(input, mask)  # features & masks, position embedding for the search
        # Adjust the shapes
        return self.adjust(features, masks, pos)

    def forward_transformer(self, feat, mask, pos):
        # Forward the transformer encoder and decoder
        output_embed, enc_mem = self.transformer(feat, mask, self.query_embed.weight,
                                                 pos, return_encoder_output=True)
        # Forward the corner head
        out, outputs_coord = self.forward_box_head(output_embed, enc_mem)
        return out, outputs_coord, output_embed

    def forward_box_head(self, hs, memory):
        """
        hs: output embeddings (1, B, N, C)
        memory: encoder embeddings (HW1 + HW2, B, C)
        """
        # adjust shape
        enc_opt = memory[-self.feat_len_s:].transpose(0, 1)  # encoder output for the search region (B, HW, C)
        dec_opt = hs.squeeze(0).transpose(1, 2)  # (B, C, N)
        att = torch.matmul(enc_opt, dec_opt)  # (B, HW, N)
        opt = (enc_opt.unsqueeze(-1) * att.unsqueeze(-2)).permute(
            (0, 3, 2, 1)).contiguous()  # (B, HW, C, N) --> (B, N, C, HW)
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)
        # run the corner head
        outputs_coord = box_xyxy_to_cxcywh(self.box_head(opt_feat))
        outputs_coord_new = outputs_coord.view(bs, Nq, 4)
        out = {'pred_boxes': outputs_coord_new}
        return out, outputs_coord_new

    def adjust(self, features: list, masks: list, pos_embed: list):
        src_feat = features[-1]
        mask = masks[-1]
        assert mask is not None
        # reduce channel
        feat = self.bottleneck(src_feat)  # (B, C, H, W)
        # adjust shapes
        feat_vec = feat.flatten(2).permute(2, 0, 1)  # HWxBxC
        pos_embed_vec = pos_embed[-1].flatten(2).permute(2, 0, 1)  # HWxBxC
        mask_vec = mask.flatten(1)  # BxHW
        return feat_vec, mask_vec, pos_embed_vec

    @torch.jit.unused
    def _set_aux_loss(self, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_boxes': b}
                for b in outputs_coord[:-1]]


def build_starks(cfg):
    backbone = build_backbone(cfg)  # backbone and positional encoding are built together
    transformer = build_transformer(cfg)
    box_head = build_box_head(cfg)
    model = STARKS(
        backbone,
        transformer,
        box_head,
        num_queries=cfg.MODEL.NUM_OBJECT_QUERIES,
        aux_loss=cfg.TRAIN.DEEP_SUPERVISION,
        head_type=cfg.MODEL.HEAD_TYPE
    )

    return model
