from models.stark.test_utils import BaseCase


class ExportStarkSTest(BaseCase):
    def test_load(self):
        original, portable = self.get_eval_models()
        self.assertIsNotNone(original)
        self.assertIsNotNone(portable)

    def test_backbone(self):
        original, portable = self.get_eval_models()
        portable.load_state_dict(original.state_dict())
        v_o = original.forward_backbone(self.inputs.backbone('original'))
        v_p = portable.forward_backbone(*self.inputs.backbone('portable'))
        self.assertLessEqual(self.diff(v_o['feat'], v_p[0]), 0, 1e-5)
        self.assertLessEqual(self.diff(v_o['mask'], v_p[1]), 0, 1e-5)
        self.assertLessEqual(self.diff(v_o['pos'], v_p[2]), 0, 1e-5)

    def test_transformer(self):
        original, portable = self.get_eval_models()
        portable.load_state_dict(original.state_dict())
        out_o, coord_o, embed_o = original.forward_transformer(self.inputs.transformer('original'))
        out_p, coord_p, embed_p = portable.forward_transformer(*self.inputs.transformer('portable'))
        self.assertLessEqual(self.diff(out_o['pred_boxes'], out_p['pred_boxes']), 1e-5)
        self.assertLessEqual(self.diff(coord_o, coord_p), 1e-5)
        self.assertLessEqual(self.diff(embed_o, embed_p), 1e-5)

    def test_validation(self):
        original, portable = self.get_eval_models()
        portable.load_state_dict(original.state_dict())
