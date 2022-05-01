import torch.nn as nn

from export import test_utils
from export.convertable import attention


def build_tf_attention(param):
    return attention.MultiheadAttention(**make_kwargs(param))


def make_kwargs(param):
    return {
        'embed_dim': param.MODEL.HIDDEN_DIM,
        'num_heads': param.MODEL.TRANSFORMER.NHEADS,
        'dropout': param.MODEL.TRANSFORMER.DROPOUT,
    }


def build_torch_attention(param):
    return nn.MultiheadAttention(**make_kwargs(param))


class AttentionTest(test_utils.DualModelTest):
    def setUp(self) -> None:
        super(AttentionTest, self).setUp()
        self.ref_builder = build_torch_attention
        self.src_builder = build_tf_attention
        self.epsilon = 1e-2

    def test_load(self):
        torch_model, tf_model = self.get_models()
        self.assertIsNotNone(torch_model)
        self.assertIsNotNone(tf_model)

    def test_inference(self):
        _, tf_model = self.get_models()
        q, k, v, key_padding_mask = self.inputs.attention(dtype=test_utils.numpy)
        v_c = tf_model((q, k, v), key_padding_mask=key_padding_mask)
        self.assertIsNotNone(v_c)

    def test_validation(self):
        torch_model, tf_model = self.get_models()
        q, k, v, key_padding_mask = self.inputs.attention(dtype=test_utils.simple)
        v_p = torch_model(q, k, v, key_padding_mask=key_padding_mask)
        q, k, v, key_padding_mask = self.inputs.attention(dtype=test_utils.numpy)
        tf_model((q, k, v), key_padding_mask=key_padding_mask)
        tf_model.import_torch_model(torch_model)
        v_c = tf_model((q, k, v), key_padding_mask=key_padding_mask)
        for p, c in zip(v_p, v_c):
            self.assertLessEqual(self.diff(p, c), self.epsilon)
