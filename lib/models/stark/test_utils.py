from unittest import TestCase

import numpy as np
import torch
from export.portable.stark_s import build_starks as portable_stark
from lib.models.stark import build_starks as stark

from lib.utils.misc import NestedTensor
from test.parameter.stark_s import parameters


class _SearchRegion:
    def __init__(self):
        self.region = np.random.random((1, 320, 320, 3)).astype(np.float32)
        self.mask = np.random.random((1, 320, 320)) < 0.5

    def astype(self, dtype):
        region, mask = torch.tensor(self.region.transpose((0, 3, 1, 2))), torch.tensor(self.mask)
        if dtype == 'portable':
            return region, mask

        if dtype == 'original':
            return NestedTensor(region, mask)

        return self.region, self.mask


class _XDict:
    def __init__(self):
        self.feat = np.random.random((400, 1, 256)).astype(np.float32)
        self.mask = np.random.random((1, 400)) < 0.5
        self.pos = np.random.randint(0, 2, (400, 1, 256))


class _ZDict:
    def __init__(self):
        self.feat = np.random.random((64, 1, 256)).astype(np.float32)
        self.mask = np.random.random((1, 64)) < 0.5
        self.pos = np.random.randint(0, 2, (64, 1, 256))


class _SeqDict:
    def __init__(self):
        self.feat = np.random.random((464, 1, 256)).astype(np.float32)
        self.mask = np.random.random((1, 464)) < 0.5
        self.pos = np.random.randint(0, 2, (464, 1, 256))

    def astype(self, dtype):
        if dtype == 'ndarray':
            return self.feat, self.mask, self.pos

        feat, mask, pos = torch.tensor(self.feat), torch.tensor(self.mask), torch.tensor(self.pos)
        if dtype == 'original':
            return {'feat': feat, 'mask': mask, 'pos': pos}

        return feat, mask, pos


class TestArray:
    def __init__(self):
        self.target_img = _SearchRegion()
        self.x_dict = _XDict()
        self.z_dict = _ZDict()
        self.seq_dict = _SeqDict()
        self.output_embed = np.random.random((1, 1, 1, 256))
        self.enc_mem = np.random.random((464, 1, 256))
        self.output_coord = np.random.random((1, 1, 4))
        self.pred_box = np.random.random((1, 1, 4))

    def backbone(self, dtype='ndarray'):
        return self.target_img.astype(dtype)

    def transformer(self, dtype='ndarray'):
        return self.seq_dict.astype(dtype)

    def template(self):
        return [torch.tensor(x) for x in [self.z_dict.feat, self.z_dict.mask, self.z_dict.pos]]


class BaseCase(TestCase):
    def setUp(self) -> None:
        self.params = parameters('baseline').cfg
        self.inputs = TestArray()

    def get_eval_models(self):
        original = stark(self.params)
        original.eval()
        portable = portable_stark(self.params)
        portable.eval()
        return original, portable

    def diff(self, a, b):
        if a.dtype == torch.bool:
            diff_mat = (a ^ b).detach().numpy()
        else:
            diff_mat = torch.abs(a - b).detach().numpy()
        return np.sum(diff_mat)
