from export import test_utils
from export.convertable.box_head import build_box_head as build_tf
from export.portable.box_head import build_box_head as build_simple


class BoxHeadTest(test_utils.DualModelTest):
    def setUp(self) -> None:
        super(BoxHeadTest, self).setUp()
        self.ref_builder = build_simple
        self.src_builder = build_tf
        self.epsilon = 1e-5

    def test_load(self):
        torch_model, tf_model = self.get_models()
        self.assertIsNotNone(torch_model)
        self.assertIsNotNone(tf_model)

    def test_validate(self):
        torch_model, tf_model = self.get_models()
        tf_model(self.inputs.box_prediction(dtype=test_utils.numpy))
        box_p = torch_model(self.inputs.box_prediction(dtype=test_utils.simple))
        tf_model.import_torch_model(torch_model)
        box_c = tf_model(self.inputs.box_prediction(dtype=test_utils.numpy))
        self.diff_inside(box_p, box_c, self.epsilon)
