from typing import Optional, Callable

import tensorflow as tf
import tensorflow.keras as tk

from export.convertable.torch_to_tf import get_conv_weights, get_bn_weights


class FrozenBatchNormalization(tk.layers.Layer):
    weight: Optional[tf.Variable]
    bias: Optional[tf.Variable]
    running_mean: Optional[tf.Variable]
    running_var: Optional[tf.Variable]

    def __init__(self, epsilon=1e-5, **kwargs):
        super(FrozenBatchNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.weight = self.add_weight(name='weight', shape=input_shape[-1:], trainable=False)
        self.bias = self.add_weight(name='bias', shape=input_shape[-1:], trainable=False)
        self.running_mean = self.add_weight(name='running_mean', shape=input_shape[-1:], trainable=False)
        self.running_var = self.add_weight(name='running_var', shape=input_shape[-1:], trainable=False)

    def call(self, x):
        scale = self.weight * tf.math.rsqrt(self.running_var + self.epsilon)
        shift = self.bias - self.running_mean * scale
        return x * scale + shift

    def compute_output_shape(self, input_shape):
        return input_shape


class Bottleneck(tk.Model):
    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: bool = False,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., tk.layers.Layer]] = None,
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = tk.layers.LayerNormalization
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = tk.layers.Conv2D(width, 1, use_bias=False)
        self.bn1 = norm_layer(epsilon=1e-5)
        self.pad = tk.layers.ZeroPadding2D(dilation)
        self.conv2 = tk.layers.Conv2D(width, 3, strides=stride, groups=groups,
                                      dilation_rate=dilation, use_bias=False)
        self.bn2 = norm_layer(epsilon=1e-5)
        self.conv3 = tk.layers.Conv2D(planes * self.expansion, 1, use_bias=False)
        self.bn3 = norm_layer(epsilon=1e-5)
        self.relu = tk.activations.relu
        self.downsample = None
        if downsample:
            self.downsample = tk.Sequential([
                tk.layers.Conv2D(planes * self.expansion, 1, strides=stride, use_bias=False),
                FrozenBatchNormalization()
            ])
        self.stride = stride

    def call(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.pad(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

    def import_torch_model(self, model):
        for name in ['conv1', 'conv2', 'conv3']:
            tf_layer = getattr(self, name)
            torch_layer = getattr(model, name)
            torch_weights = get_conv_weights(torch_layer)
            tf_layer.set_weights(torch_weights)
        for name in ['bn1', 'bn2', 'bn3']:
            tf_layer = getattr(self, name)
            torch_layer = getattr(model, name)
            tf_layer.set_weights(get_bn_weights(torch_layer))
        if model.downsample is not None:
            down_layer = model.downsample
            self.downsample.set_weights(
                get_conv_weights(down_layer[0]) + get_bn_weights(down_layer[1])
            )


class L3TruncatedResNet(tk.Model):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, last_layer=None):
        super(L3TruncatedResNet, self).__init__()
        self.last_layer = last_layer
        self._norm_layer = FrozenBatchNormalization

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.pad1_conv = tk.layers.ZeroPadding2D(3)
        self.conv1 = tk.layers.Conv2D(self.inplanes, kernel_size=7, strides=2,
                                      use_bias=False)
        self.bn1 = norm_layer()
        self.relu = tk.layers.ReLU()
        self.pad1_pool = tk.layers.ZeroPadding2D(1)
        self.maxpool = tk.layers.MaxPool2D(pool_size=3, strides=2)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        last_channel = 256
        self.layer3 = self._make_layer(block, last_channel, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.num_channels = last_channel * block.expansion

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = False
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = True

        layers = [
            block(self.inplanes, planes, stride, downsample,
                  self.groups, self.base_width, previous_dilation, norm_layer)
        ]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
        return tk.Sequential(layers)

    def call(self, x):
        x = self.pad1_conv(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pad1_pool(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x

    def import_torch_model(self, model):
        self.conv1.set_weights(get_conv_weights(model.conv1))
        self.bn1.set_weights(get_bn_weights(model.bn1))
        for l_name in ['layer1', 'layer2', 'layer3']:
            for tf_btn, torch_btn in zip(getattr(self, l_name).layers, getattr(model, l_name)):
                tf_btn.import_torch_model(torch_btn)


def _resnet(block, layers, **kwargs):
    return L3TruncatedResNet(block, layers, **kwargs)


def resnet50(**kwargs):
    return _resnet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    return _resnet(Bottleneck, [3, 4, 23, 3], **kwargs)
