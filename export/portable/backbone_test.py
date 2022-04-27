from export.portable.backbone import Backbone as BackboneP
from lib.models.stark.backbone import Backbone as BackboneO
from models.stark.test_utils import BaseCase


class BackboneTest(BaseCase):
    def testBackbone(self):
        train_backbone = self.params.TRAIN.BACKBONE_MULTIPLIER > 0
        return_interm_layers = self.params.MODEL.PREDICT_MASK
        bb_original = BackboneO(self.params.MODEL.BACKBONE.TYPE, train_backbone, return_interm_layers,
                                self.params.MODEL.BACKBONE.DILATION, self.params.TRAIN.FREEZE_BACKBONE_BN, None)

        bb_portable = BackboneP(self.params.MODEL.BACKBONE.TYPE, train_backbone, return_interm_layers,
                                self.params.MODEL.BACKBONE.DILATION, self.params.TRAIN.FREEZE_BACKBONE_BN, None)

        tensor_dict = bb_original(self.inputs.backbone('original'))
        xs_o = [tensor.tensors for tensor in tensor_dict.values()]
        masks_o = [tensor.mask for tensor in tensor_dict.values()]
        xs_p, masks_p = bb_portable(*self.inputs.backbone('portable'))

        for o, p in zip(xs_o, xs_p):
            self.assertLessEqual(self.diff(o, p), 0, 1e-5)
        for o, p in zip(masks_o, masks_p):
            self.assertLessEqual(self.diff(o, p), 0, 1e-5)
