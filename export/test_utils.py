from typing import Callable
from unittest import TestCase

import numpy as np
import tensorflow
import torch

from lib.test.parameter.stark_s import parameters
from lib.utils.misc import NestedTensor

__all__ = ['origin', 'numpy', 'simple', 'tf', 'DualModelTest']
origin = 0
numpy = 1
simple = 2
tf = 3


def _channel_last_to_first(array):
    return array.transpose((0, 3, 1, 2))


class _TestArray:
    def __init__(self):
        self.target_img = np.random.random((1, 320, 320, 3)).astype(np.float32)
        self.img_mask = np.random.random((1, 320, 320)) < 0.5

        self.backbone_feature = np.random.random((1, 20, 20, 1024)).astype(np.float32)
        self.backbone_mask = np.random.random((1, 20, 20)) < 0.5

        self.feature = np.random.random((464, 1, 256)).astype(np.float32)
        self.mask = np.random.random((1, 464)) < 0.5
        self.query = np.random.random((1, 256)).astype(np.float32)
        self.pos = np.random.randint(0, 2, (464, 1, 256)).astype(np.float32)

        self.q_k = np.random.random((464, 1, 256)).astype(np.float32)
        self.feature_mask = self.feature < 0.5

        self.emb_feat = np.random.random((1, 20, 20, 256)).astype(np.float32)

    def backbone(self, dtype=numpy, cuda=False):
        if dtype == numpy:
            return self.target_img, self.img_mask

        target_img = torch.tensor(_channel_last_to_first(self.target_img))
        mask = torch.tensor(self.img_mask)
        if cuda:
            target_img = target_img.cuda()
            mask = mask.cuda()
        if dtype == origin:
            return NestedTensor(target_img, mask)

        return target_img, mask

    def position(self, dtype=numpy):
        if dtype == numpy:
            return self.backbone_feature, self.backbone_mask

        feature = torch.tensor(_channel_last_to_first(self.backbone_feature))
        mask = torch.tensor(self.backbone_mask)
        if dtype == origin:
            return NestedTensor(feature, mask)

        return feature, mask

    def transformer(self, dtype=numpy, cuda=False):
        if dtype == numpy:
            return self.feature, self.mask, self.pos

        feature, mask, pos = (torch.tensor(x) for x in [self.feature, self.mask, self.pos])
        if cuda:
            feature = feature.cuda()
            mask = mask.cuda()
            pos = pos.cuda()
        if dtype == origin:
            return {'feat': feature, 'mask': mask, 'pos': pos}
        return feature, mask, pos

    def transformer_net(self, dtype=numpy):
        feature, mask, pos = self.transformer(dtype=dtype)

        query = self.query
        if dtype != numpy:
            query = torch.tensor(self.query)

        return feature, mask, query, pos

    def attention(self, dtype=numpy):
        outputs = [self.q_k, self.q_k, self.feature, self.mask]
        if dtype == numpy:
            return outputs

        return [torch.tensor(x) for x in outputs]

    def box_prediction(self, dtype=numpy):
        if dtype == numpy:
            return self.emb_feat

        return torch.tensor(_channel_last_to_first(self.emb_feat))


def _apply(fn):
    def apply(*args):
        return fn(*args)

    return apply


class DualModelTest(TestCase):
    ref_builder: Callable
    src_builder: Callable

    def setUp(self) -> None:
        self.params = parameters('baseline').cfg
        self.inputs = _TestArray()

    def __setattr__(self, key, value):
        if key in ('ref_builder', 'src_builder'):
            value = _apply(value)
        super(DualModelTest, self).__setattr__(key, value)

    def get_models(self):
        if not hasattr(self, 'ref_builder'):
            raise AttributeError('self.ref_builder must be assigned in setUp()')
        if not hasattr(self, 'src_builder'):
            raise AttributeError('self.src_builder must be assigned in setUp()')
        ref = self.ref_builder(self.params)
        src = self.src_builder(self.params)
        if isinstance(ref, torch.nn.Module):
            ref.eval()
        if isinstance(src, torch.nn.Module):
            src.eval()
        return ref, src

    def get_copied_models(self):
        ref, src = self.get_models()
        if not isinstance(ref, torch.nn.Module):
            raise AttributeError('Reference should be an object of torch.nn.Module')
        if isinstance(src, torch.nn.Module):
            src.load_state_dict(ref.state_dict())
        else:
            src.import_torch_model(ref)
        return ref, src

    def _diff(self, a, b, channel_align=False):
        a = _to_numpy(a, channel_align)
        b = _to_numpy(b, channel_align)
        if a.dtype == bool:
            diff_arr = (a ^ b).astype(int)
        else:
            diff_arr = np.abs(a - b)
        return diff_arr.max()

    def diff_inside(self, a, b, epsilon=1e-5, channel_align=False):
        diff_value = self._diff(a, b, channel_align=channel_align)
        print(self.__class__, diff_value)
        self.assertLessEqual(diff_value, epsilon)

    def get_tf_converted(self, model):
        converter = tensorflow.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tensorflow.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tensorflow.float16]
        flat_buffer = converter.convert()
        tflite_interpreter = tensorflow.lite.Interpreter(model_content=flat_buffer)
        tflite_interpreter.allocate_tensors()
        input_details = tflite_interpreter.get_input_details()
        output_details = tflite_interpreter.get_output_details()

        class_name = str(type(model).__name__)
        for index, input_detail in enumerate(input_details):
            print("{} input({}) - name: {}, shape: {}, type: {}".format(
                class_name, index, input_detail['name'], input_detail['shape'], input_detail['dtype'])
            )

        for index, output_detail in enumerate(output_details):
            print("{} output({}) - name: {}, shape: {}, type: {}".format(
                class_name, index, output_detail['name'], output_detail['shape'], output_detail['dtype'])
            )
        return tflite_interpreter

    def call_tflite_model(self, tflite_model: tensorflow.lite.Interpreter, *args):
        for input_detail, input_tensor in zip(tflite_model.get_input_details(), args):
            tflite_model.set_tensor(input_detail['index'], input_tensor)
        tflite_model.invoke()
        return [tflite_model.get_tensor(output_detail['index']) for output_detail
                in tflite_model.get_output_details()]


def _to_numpy(tensor, channel_align):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().numpy()
        if channel_align:
            tensor = tensor.transpose((0, 2, 3, 1))
    elif isinstance(tensor, tensorflow.Tensor):
        tensor = tensor.numpy()
    return tensor
