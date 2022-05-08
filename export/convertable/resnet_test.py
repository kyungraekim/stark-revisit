import torch.nn as nn
from torchvision.models.resnet import Bottleneck as tc_btn

from export import test_utils
from export.convertable.resnet import Bottleneck as tf_btn, FrozenBatchNormalization
from export.convertable.resnet import resnet50 as tf_resnet50
from export.portable.backbone import FrozenBatchNorm2d
from export.portable.resnet import resnet50 as torch_resnet50


def _bottleneck_params():
    return {
        'inplanes': 3,
        'planes': 64,
        'stride': 1,
        'groups': 1,
        'base_width': 64,
        'dilation': 1,
    }


def build_torch_bottleneck(*args):
    params = _bottleneck_params()
    out_channels = params['planes'] * tc_btn.expansion
    params['downsample'] = nn.Sequential(
        nn.Conv2d(params['inplanes'], out_channels, 1, stride=params['stride'], bias=False),
        FrozenBatchNorm2d(out_channels)
    )
    params['norm_layer'] = FrozenBatchNorm2d
    return tc_btn(**params)


def build_tf_bottleneck(*args):
    params = _bottleneck_params()
    params['downsample'] = True
    params['norm_layer'] = FrozenBatchNormalization
    return tf_btn(**params)


class BottleNeckTest(test_utils.DualModelTest):
    def setUp(self) -> None:
        super(BottleNeckTest, self).setUp()
        self.ref_builder = build_torch_bottleneck
        self.src_builder = build_tf_bottleneck
        self.epsilon = 1e-4

    def test_load(self):
        torch_model, tf_model = self.get_models()
        self.assertIsNotNone(torch_model)
        self.assertIsNotNone(tf_model)

    def test_inference(self):
        _, tf_model = self.get_models()
        sample_img, _ = self.inputs.backbone(dtype=test_utils.numpy)
        self.assertIsNotNone(tf_model(sample_img))

    def test_validation(self):
        torch_model, tf_model = self.get_models()
        sample_img, _ = self.inputs.backbone(dtype=test_utils.simple)
        v_p = torch_model(sample_img)
        sample_img, _ = self.inputs.backbone(dtype=test_utils.numpy)
        tf_model(sample_img)
        tf_model.import_torch_model(torch_model)
        v_c = tf_model(sample_img)
        self.diff_inside(v_p, v_c, channel_align=True, epsilon=self.epsilon)


def _resnet_param(norm_layer):
    return {
        'replace_stride_with_dilation': [False, False, False],
        'last_layer': 'layer3',
        'norm_layer': norm_layer,
    }


def build_torch_resnet(*args):
    return torch_resnet50(**_resnet_param(FrozenBatchNorm2d))


def build_tf_resnet(*args):
    return tf_resnet50(**_resnet_param(FrozenBatchNormalization))


class ResNetTest(test_utils.DualModelTest):
    def setUp(self) -> None:
        super(ResNetTest, self).setUp()
        self.ref_builder = build_torch_resnet
        self.src_builder = build_tf_resnet
        self.epsilon = 1e-3

    def test_load(self):
        torch_model, tf_model = self.get_models()
        self.assertIsNotNone(torch_model)
        self.assertIsNotNone(tf_model)

    def test_inference(self):
        _, tf_model = self.get_models()
        sample_img, _ = self.inputs.backbone(dtype=test_utils.numpy)
        self.assertIsNotNone(tf_model(sample_img))

    def test_validation(self):
        torch_model, tf_model = self.get_models()
        sample_img, _ = self.inputs.backbone(dtype=test_utils.simple)
        v_p = torch_model(sample_img)
        sample_img, _ = self.inputs.backbone(dtype=test_utils.numpy)
        tf_model(sample_img)
        tf_model.import_torch_model(torch_model)
        v_c = tf_model(sample_img)
        self.diff_inside(v_p, v_c, channel_align=True, epsilon=self.epsilon)

    def test_tflite(self):
        _, tf_model = self.get_models()
        tf_model(self.inputs.backbone(dtype=test_utils.numpy)[0])
        tflite_model = self.get_tf_converted(tf_model)
        self.assertIsNotNone(tflite_model)

    def test_torch_to_tflite(self):
        torch_model, tf_model = self.get_models()
        sample_img, _ = self.inputs.backbone(dtype=test_utils.simple)
        sample_img, _ = self.inputs.backbone(dtype=test_utils.numpy)
        tf_model(sample_img)
        tf_model.import_torch_model(torch_model)
        v_c = tf_model(sample_img)
        tflite_model = self.get_tf_converted(tf_model)
        self.assertIsNotNone(tflite_model)

        v_lite = self.call_tflite_model(tflite_model, sample_img)[0]
        self.diff_inside(v_c, v_lite, epsilon=1e-1)
