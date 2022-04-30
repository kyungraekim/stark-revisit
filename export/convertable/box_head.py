import tensorflow as tf
import tensorflow.keras as tk


def conv(out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return tk.Sequential([
        tk.layers.ZeroPadding2D(padding=padding),
        tk.layers.Conv2D(out_planes, kernel_size=kernel_size, strides=stride, dilation_rate=dilation,
                         use_bias=True),
        tk.layers.BatchNormalization(epsilon=1e-5, momentum=0.9),
        tk.layers.ReLU()
    ])


class CornerPredictor(tk.Model):
    """ Corner Predictor module"""

    def __init__(self, inplanes=64, channel=256, feat_sz=20, stride=16):
        super(CornerPredictor, self).__init__()
        self.feat_sz = feat_sz
        self.stride = stride
        self.img_sz = self.feat_sz * self.stride
        '''top-left corner'''
        self.conv1_tl = conv(channel)
        self.conv2_tl = conv(channel // 2)
        self.conv3_tl = conv(channel // 4)
        self.conv4_tl = conv(channel // 8)
        self.conv5_tl = tk.layers.Conv2D(1, kernel_size=1)

        '''bottom-right corner'''
        self.conv1_br = conv(channel)
        self.conv2_br = conv(channel // 2)
        self.conv3_br = conv(channel // 4)
        self.conv4_br = conv(channel // 8)
        self.conv5_br = tk.layers.Conv2D(1, kernel_size=1)

        '''about coordinates and indexs'''
        self.indice = tf.cast(tf.reshape(tf.range(0, self.feat_sz), (-1, 1)) * self.stride, dtype=tf.float32)
        self.coord_x = tf.reshape(tf.tile(self.indice, (self.feat_sz, 1)), (self.feat_sz * self.feat_sz,))
        self.coord_y = tf.reshape(tf.tile(self.indice, (1, self.feat_sz)), (self.feat_sz * self.feat_sz,))

    def call(self, x, return_dist=False, softmax=True):
        score_map_tl, score_map_br = self.get_score_map(x)
        if return_dist:
            coorx_tl, coory_tl, prob_vec_tl = self.soft_argmax(score_map_tl, return_dist=True, softmax=softmax)
            coorx_br, coory_br, prob_vec_br = self.soft_argmax(score_map_br, return_dist=True, softmax=softmax)
            return tf.stack((coorx_tl, coory_tl, coorx_br, coory_br), axis=1) / self.img_sz, prob_vec_tl, prob_vec_br
        else:
            coorx_tl, coory_tl = self.soft_argmax(score_map_tl)
            coorx_br, coory_br = self.soft_argmax(score_map_br)
            return tf.stack((coorx_tl, coory_tl, coorx_br, coory_br), axis=1) / self.img_sz

    def get_score_map(self, x):
        # top-left branch
        x_tl1 = self.conv1_tl(x)
        x_tl2 = self.conv2_tl(x_tl1)
        x_tl3 = self.conv3_tl(x_tl2)
        x_tl4 = self.conv4_tl(x_tl3)
        score_map_tl = self.conv5_tl(x_tl4)

        # bottom-right branch
        x_br1 = self.conv1_br(x)
        x_br2 = self.conv2_br(x_br1)
        x_br3 = self.conv3_br(x_br2)
        x_br4 = self.conv4_br(x_br3)
        score_map_br = self.conv5_br(x_br4)
        return score_map_tl, score_map_br

    def soft_argmax(self, score_map, return_dist=False, softmax=True):
        """ get soft-argmax coordinate for a given heatmap """
        score_vec = tf.reshape(score_map, (-1, self.feat_sz * self.feat_sz))  # (batch, feat_sz * feat_sz)
        prob_vec = tf.math.softmax(score_vec, axis=1)
        exp_x = tf.math.reduce_sum((self.coord_x * prob_vec), axis=1)
        exp_y = tf.math.reduce_sum((self.coord_y * prob_vec), axis=1)
        if return_dist:
            if softmax:
                return exp_x, exp_y, prob_vec
            else:
                return exp_x, exp_y, score_vec
        else:
            return exp_x, exp_y

    def import_torch_model(self, model):
        name_format = 'conv{}_{}'
        for postfix in ['br', 'tl']:
            for i in range(1, 5):
                name = name_format.format(i, postfix)
                tf_layer = getattr(self, name)
                assert isinstance(tf_layer, tk.Sequential)
                torch_layer = getattr(model, name)
                tf_layer.set_weights(get_seq_weights(torch_layer))

            name = name_format.format(5, postfix)
            tf_layer = getattr(self, name)
            assert isinstance(tf_layer, tk.layers.Conv2D)
            torch_layer = getattr(model, name)
            tf_layer.set_weights(get_conv_weights(torch_layer))


def get_seq_weights(layer):
    conv_weights = get_conv_weights(layer[0])
    bn_weights = get_bn_weights(layer[1])
    return conv_weights + bn_weights


def get_weights(layer, names):
    return [getattr(layer, name).detach().numpy() for name in names]


def get_conv_weights(layer):
    weights = get_weights(layer, ['weight', 'bias'])
    weights[0] = weights[0].transpose((2, 3, 1, 0))
    return weights


def get_bn_weights(layer):
    return get_weights(layer, ['weight', 'bias', 'running_mean', 'running_var'])


def build_box_head(cfg):
    if cfg.MODEL.BACKBONE.DILATION is False:
        stride = 16
    else:
        stride = 8
    feat_sz = int(cfg.DATA.SEARCH.SIZE / stride)
    channel = getattr(cfg.MODEL, "HEAD_DIM", 256)
    print("head channel: %d" % channel)
    corner_head = CornerPredictor(inplanes=cfg.MODEL.HIDDEN_DIM, channel=channel,
                                  feat_sz=feat_sz, stride=stride)
    return corner_head
