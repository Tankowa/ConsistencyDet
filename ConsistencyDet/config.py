
from detectron2.config import CfgNode as CN


def add_ConsistencyDet_config(cfg):
    """
    Add config for ConsistencyDet
    """
    cfg.MODEL.ConsistencyDet = CN()
    cfg.MODEL.ConsistencyDet.NUM_CLASSES = 80
    cfg.MODEL.ConsistencyDet.NUM_PROPOSALS = 300

    # Consistency Model
    cfg.MODEL.ConsistencyDet.sigma_max = 40
    cfg.MODEL.ConsistencyDet.sigma_min = 0.002
    cfg.MODEL.ConsistencyDet.sigma_data = 0.5
    cfg.MODEL.ConsistencyDet.rho = 7
    cfg.MODEL.ConsistencyDet.n_steps = 40
    cfg.MODEL.ConsistencyDet.teacher_path = "/diffdet_coco_res50.pth"

    # RCNN Head.
    cfg.MODEL.ConsistencyDet.NHEADS = 8
    cfg.MODEL.ConsistencyDet.DROPOUT = 0.0
    cfg.MODEL.ConsistencyDet.DIM_FEEDFORWARD = 2048
    cfg.MODEL.ConsistencyDet.ACTIVATION = 'relu'
    cfg.MODEL.ConsistencyDet.HIDDEN_DIM = 256
    cfg.MODEL.ConsistencyDet.NUM_CLS = 1
    cfg.MODEL.ConsistencyDet.NUM_REG = 3
    cfg.MODEL.ConsistencyDet.NUM_HEADS = 6

    # Dynamic Conv.
    cfg.MODEL.ConsistencyDet.NUM_DYNAMIC = 2
    cfg.MODEL.ConsistencyDet.DIM_DYNAMIC = 64

    # Loss.
    cfg.MODEL.ConsistencyDet.CLASS_WEIGHT = 2.0
    cfg.MODEL.ConsistencyDet.GIOU_WEIGHT = 2.0
    cfg.MODEL.ConsistencyDet.L1_WEIGHT = 5.0
    cfg.MODEL.ConsistencyDet.CONSISTENCY_WEIGHT = 2.0
    cfg.MODEL.ConsistencyDet.DEEP_SUPERVISION = True
    cfg.MODEL.ConsistencyDet.NO_OBJECT_WEIGHT = 0.1

    # Focal Loss.
    cfg.MODEL.ConsistencyDet.USE_FOCAL = True
    cfg.MODEL.ConsistencyDet.USE_FED_LOSS = False
    cfg.MODEL.ConsistencyDet.ALPHA = 0.25
    cfg.MODEL.ConsistencyDet.GAMMA = 2.0
    cfg.MODEL.ConsistencyDet.PRIOR_PROB = 0.01

    # Dynamic K
    cfg.MODEL.ConsistencyDet.OTA_K = 5

    # Diffusion
    cfg.MODEL.ConsistencyDet.SNR_SCALE = 2.0
    cfg.MODEL.ConsistencyDet.SAMPLE_STEP = 1

    # Inference
    cfg.MODEL.ConsistencyDet.USE_NMS = True

    # Swin Backbones
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.SIZE = 'B'  # 'T', 'S', 'B'
    cfg.MODEL.SWIN.USE_CHECKPOINT = False
    cfg.MODEL.SWIN.OUT_FEATURES = (0, 1, 2, 3)  # modify

    # Optimizer.
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0

    # TTA.
    cfg.TEST.AUG.MIN_SIZES = (400, 500, 600, 640, 700, 900, 1000, 1100, 1200, 1300, 1400, 1800, 800)
    cfg.TEST.AUG.CVPODS_TTA = True
    cfg.TEST.AUG.SCALE_FILTER = True
    cfg.TEST.AUG.SCALE_RANGES = ([96, 10000], [96, 10000], 
                                 [64, 10000], [64, 10000],
                                 [64, 10000], [0, 10000],
                                 [0, 10000], [0, 256],
                                 [0, 256], [0, 192],
                                 [0, 192], [0, 96],
                                 [0, 10000])
