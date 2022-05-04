from export import test_utils
from export.convertable.position_encoding import build_position_encoding as tf_encoder
from export.portable.position_encoding import build_position_encoding as torch_encoder


class PositionEncodingTest(test_utils.DualModelTest):
    def setUp(self) -> None:
        super(PositionEncodingTest, self).setUp()
        self.ref_builder = torch_encoder
        self.src_builder = tf_encoder
        self.epsilon = 1e-6

    def test_load(self):
        torch_model, tf_model = self.get_models()
        self.assertIsNotNone(torch_model)
        self.assertIsNotNone(tf_model)

    def test_inference(self):
        _, tf_model = self.get_models()
        _, position = self.inputs.position(dtype=test_utils.numpy)
        self.assertIsNotNone(tf_model(position))

    def test_validation(self):
        torch_model, tf_model = self.get_models()
        v_p = torch_model(*self.inputs.position(dtype=test_utils.simple))
        _, position = self.inputs.position(dtype=test_utils.numpy)
        tf_model(position)
        tf_model.import_torch_model(torch_model)
        v_c = tf_model(position)
        self.diff_inside(v_p, v_c, channel_align=True, epsilon=self.epsilon)
