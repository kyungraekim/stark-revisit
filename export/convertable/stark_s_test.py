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
        self.diff_inside(tc_output[1], tf_output)

    def test_tflite_extractor(self):
        _, tf_model = self.get_models()
        tf_model = tf_model.feature_extractor
        tf_model(self.inputs.backbone(dtype=test_utils.numpy))
        tflite_model = self.get_tf_converted(tf_model)
        self.assertIsNotNone(tflite_model)

    def test_torch_to_tflite_extractor(self):
        self._test_torch_to_tflite('feature_extractor', self.inputs.backbone)

    def test_tflite_detector(self):
        _, tf_model = self.get_models()
        tf_model = tf_model.detector
        tf_model(self.inputs.transformer(dtype=test_utils.numpy))
        tflite_model = self.get_tf_converted(tf_model)
        self.assertIsNotNone(tflite_model)

    def test_torch_to_tflite_extractor(self):
        self._test_torch_to_tflite('detector', self.inputs.transformer)

    def _test_torch_to_tflite(self, attr, input_source):
        tc_model, tf_model = self.get_models()
        tf_model = getattr(tf_model, attr)
        tf_input = input_source(dtype=test_utils.numpy)
        tf_model(tf_input)
        tf_model.import_torch_model(tc_model)
        v_c = tf_model(tf_input)
        tflite_model = self.get_tf_converted(tf_model)
        self.assertIsNotNone(tflite_model)

        v_lite = self.call_tflite_model(tflite_model, *tf_input)
        for tf_v, tflite_v in zip(v_c, v_lite):
            self.diff_inside(tf_v, tflite_v, epsilon=1e-2)
