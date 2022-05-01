from export import test_utils
from export.portable.backbone import build_backbone as build_simple
from lib.models.stark.backbone import build_backbone as build_origin


class BackboneTest(test_utils.DualModelTest):
    def setUp(self) -> None:
        super(BackboneTest, self).setUp()
        self.ref_builder = build_origin
        self.src_builder = build_simple
        self.epsilon = 1e-5

    def testBackbone(self):
        bb_original, bb_portable = self.get_copied_models()

        tensor_dict = bb_original(self.inputs.backbone(test_utils.origin))
        xs_o = [tensor.tensors for tensor in tensor_dict[0]]
        mask_o = [tensor.mask for tensor in tensor_dict[0]]
        pos_o = tensor_dict[1]
        xs_p, mask_p, pos_p = bb_portable(*self.inputs.backbone(test_utils.simple))

        for o, p in zip(xs_o, xs_p):
            self.diff_inside(o, p, self.epsilon)
        for o, p in zip(mask_o, mask_p):
            self.diff_inside(o, p, self.epsilon)
        for o, p in zip(pos_o, pos_p):
            self.diff_inside(o, p, self.epsilon)
