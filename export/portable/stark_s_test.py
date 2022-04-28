from export import test_utils
from export.portable.stark_s import build_starks as build_portable
from lib.models.stark.stark_s import build_starks as build_original


class ExportStarkSTest(test_utils.DualModelTest):
    def setUp(self) -> None:
        super(ExportStarkSTest, self).setUp()
        self.ref_builder = build_original
        self.src_builder = build_portable
        self.epsilon = 1e-5

    def test_load(self):
        original, portable = self.get_models()
        self.assertIsNotNone(original)
        self.assertIsNotNone(portable)

    def test_backbone(self):
        original, portable = self.get_copied_models()
        portable.load_state_dict(original.state_dict())
        v_o = original.forward_backbone(self.inputs.backbone(test_utils.origin))
        v_p = portable.forward_backbone(*self.inputs.backbone(test_utils.simple))
        self.assertLessEqual(self.diff(v_o['feat'], v_p[0]), self.epsilon)
        self.assertLessEqual(self.diff(v_o['mask'], v_p[1]), self.epsilon)
        self.assertLessEqual(self.diff(v_o['pos'], v_p[2]), self.epsilon)

    def test_transformer(self):
        original, portable = self.get_copied_models()
        portable.load_state_dict(original.state_dict())
        out_o, coord_o, embed_o = original.forward_transformer(self.inputs.transformer(test_utils.origin))
        out_p, coord_p, embed_p = portable.forward_transformer(*self.inputs.transformer(test_utils.simple))
        self.assertLessEqual(self.diff(out_o['pred_boxes'], out_p['pred_boxes']), self.epsilon)
        self.assertLessEqual(self.diff(coord_o, coord_p), self.epsilon)
        self.assertLessEqual(self.diff(embed_o, embed_p), self.epsilon)
