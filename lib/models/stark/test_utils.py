import numpy as np


class _SearchRegion:
    def __init__(self):
        self.region = np.random.random((1, 320, 320, 3)).astype(np.float32)
        self.mask = np.random.random((1, 320, 320)) < 0.5


class _XDict:
    def __init__(self):
        self.feat = np.random.random((400, 1, 256)).astype(np.float32)
        self.mask = np.random.random((1, 400)) < 0.5
        self.pos = np.random.randint(0, 2, (400, 1, 256))


class _SeqDict:
    def __init__(self):
        self.feat = np.random.random((464, 1, 256))
        self.mask = np.random.random((1, 464)) < 0.5
        self.pos = np.random.randint(0, 2, (464, 1, 256))


class TestArray:
    def __init__(self):
        self.target_img = _SearchRegion()
        self.x_dict = _XDict()
        self.seq_dict = _SeqDict()
        self.output_embed = np.random.random((1, 1, 1, 256))
        self.enc_mem = np.random.random((464, 1, 256))
        self.output_coord = np.random.random((1, 1, 4))
        self.pred_box = np.random.random((1, 1, 4))
