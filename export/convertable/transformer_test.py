from export import test_utils
from export.convertable.transformer import build_transformer as build_tf
from export.portable.transformer import build_transformer as build_simple


class TransformerTest(test_utils.DualModelTest):
    def setUp(self) -> None:
        super(TransformerTest, self).setUp()
        self.ref_builder = build_simple
        self.src_builder = build_tf
        self.epsilon = 1e-5

    def test_load(self):
        torch_model, tf_model = self.get_models()
        self.assertIsNotNone(torch_model)
        self.assertIsNotNone(tf_model)

    def test_inference(self):
        _, tf_model = self.get_models()
        box_c = tf_model(self.inputs.transformer_net(dtype=test_utils.numpy))
        self.assertIsNotNone(box_c)

    def test_validate(self):
        torch_model, tf_model = self.get_models()
        box_p = torch_model(
            *self.inputs.transformer_net(dtype=test_utils.simple),
            return_encoder_output=True
        )
        tf_model(self.inputs.transformer_net(dtype=test_utils.numpy))
        tf_model.import_torch_model(torch_model)
        box_c = tf_model(self.inputs.transformer_net(dtype=test_utils.numpy))
        self.diff_inside(box_p[0], box_c[0], self.epsilon)
        self.diff_inside(box_p[1], box_c[1], self.epsilon)

    def test_tflite(self):
        _, tf_model = self.get_models()
        tf_model(self.inputs.transformer_net(dtype=test_utils.numpy))
        tflite_model = self.get_tf_converted(tf_model)
        self.assertIsNotNone(tflite_model)

    def test_torch_to_tflite(self):
        torch_model, tf_model = self.get_models()
        tf_input = self.inputs.transformer_net(dtype=test_utils.numpy)
        tf_model(tf_input)
        tf_model.import_torch_model(torch_model)
        v_c = tf_model(self.inputs.transformer_net(dtype=test_utils.numpy))
        tflite_model = self.get_tf_converted(tf_model)
        self.assertIsNotNone(tflite_model)

        v_lite = self.call_tflite_model(tflite_model, *tf_input)
        self.diff_inside(v_c[0], v_lite[0], epsilon=1e-1)
        self.diff_inside(v_c[1], v_lite[1], epsilon=1e-1)
