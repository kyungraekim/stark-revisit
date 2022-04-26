class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/Users/kyungraekim/workspace/ai/stark'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/Users/kyungraekim/workspace/ai/stark/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/Users/kyungraekim/workspace/ai/stark/pretrained_networks'
        self.lasot_dir = '/Users/kyungraekim/workspace/ai/stark/data/lasot'
        self.got10k_dir = '/Users/kyungraekim/workspace/ai/stark/data/got10k'
        self.lasot_lmdb_dir = '/Users/kyungraekim/workspace/ai/stark/data/lasot_lmdb'
        self.got10k_lmdb_dir = '/Users/kyungraekim/workspace/ai/stark/data/got10k_lmdb'
        self.trackingnet_dir = '/Users/kyungraekim/workspace/ai/stark/data/trackingnet'
        self.trackingnet_lmdb_dir = '/Users/kyungraekim/workspace/ai/stark/data/trackingnet_lmdb'
        self.coco_dir = '/Users/kyungraekim/workspace/ai/stark/data/coco'
        self.coco_lmdb_dir = '/Users/kyungraekim/workspace/ai/stark/data/coco_lmdb'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = '/Users/kyungraekim/workspace/ai/stark/data/vid'
        self.imagenet_lmdb_dir = '/Users/kyungraekim/workspace/ai/stark/data/vid_lmdb'
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
