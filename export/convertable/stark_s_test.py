from export import test_utils
from export.convertable.stark_s import build_stark_s
from export.portable.stark_s import build_starks


class StarkSTest(test_utils.DualModelTest):
    def setUp(self) -> None:
        super(StarkSTest, self).setUp()
        self.ref_builder = build_starks
        self.src_builder = build_stark_s

    def test_load(self):
        tc_model, tf_model = self.get_models()
        self.assertIsNotNone(tc_model)
        self.assertIsNotNone(tf_model)

    def test_forward_backbone(self):
        tc_model, tf_model = self.get_models()
        tc_output = tc_model.forward_backbone(*self.inputs.backbone(dtype=test_utils.simple))
        tf_model.forward_backbone(*self.inputs.backbone(dtype=test_utils.numpy))
        tf_model.import_torch_feature_extractor(tc_model)
        tf_output = tf_model.forward_backbone(*self.inputs.backbone(dtype=test_utils.numpy))
        for tc_v, tf_v in zip(tc_output, tf_output):
            self.diff_inside(tc_v, tf_v)

    def test_forward_transformer(self):
        tc_model, tf_model = self.get_models()
        tc_output = tc_model.forward_transformer(*self.inputs.transformer(dtype=test_utils.simple))
        tf_model.forward_transformer(*self.inputs.transformer(dtype=test_utils.numpy))
        tf_model.import_torch_transformer(tc_model)
        tf_output = tf_model.forward_transformer(*self.inputs.transformer(dtype=test_utils.numpy))
        for tc_v, tf_v in zip(tc_output[1], tf_output):
            self.diff_inside(tc_v, tf_v)
