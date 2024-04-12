
import math
import random
from typing import List
from collections import namedtuple
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.layers import batched_nms
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, detector_postprocess

from detectron2.structures import Boxes, ImageList, Instances

from .loss import SetCriterionDynamicK, HungarianMatcherDynamicK
from .head import DynamicHead
from .util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from .util.misc import nested_tensor_from_tensor_list
from .detection_decoder import DynamicHead as teacher_model

__all__ = ["ConsistencyDet"]

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])


class DummyGenerator:
    def randn(self, *args, **kwargs):
        return torch.randn(*args, **kwargs)

    def randint(self, *args, **kwargs):
        return torch.randint(*args, **kwargs)

    def randn_like(self, *args, **kwargs):
        return torch.randn_like(*args, **kwargs)


def get_generator(generator, num_samples=0, seed=0):
    if generator == "dummy":
        return DummyGenerator()


def get_sigmas_karras(sigma_min, sigma_max, rho, n_steps):
    """Constructs the noise schedule of Karras et al. (2022)."""
    ramp = torch.linspace(0, 1, n_steps)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas)


def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]


def get_snr(sigmas):
    return sigmas ** -2


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def extract(a, t, x_shape):
    """extract the appropriate  t  index for a batch of indices"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


@META_ARCH_REGISTRY.register()
class ConsistencyDet(nn.Module):
    """
    Implement ConsistencyDet
    """

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)
        self.batch = cfg.SOLVER.IMS_PER_BATCH
        self.in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.num_classes = cfg.MODEL.ConsistencyDet.NUM_CLASSES
        self.num_proposals = cfg.MODEL.ConsistencyDet.NUM_PROPOSALS
        self.hidden_dim = cfg.MODEL.ConsistencyDet.HIDDEN_DIM
        self.num_heads = cfg.MODEL.ConsistencyDet.NUM_HEADS

        self.heun_solver = True
        # Build Backbone.
        self.backbone = build_backbone(cfg)
        self.size_divisibility = self.backbone.size_divisibility

        # Build Teacher Model for Distillation
        self.teacher_model = None

        # Build consistency parameters
        self.sigma_max = cfg.MODEL.ConsistencyDet.sigma_max
        self.sigma_min = cfg.MODEL.ConsistencyDet.sigma_min
        self.sigma_data = cfg.MODEL.ConsistencyDet.sigma_data
        self.rho = cfg.MODEL.ConsistencyDet.rho
        self.n_steps = cfg.MODEL.ConsistencyDet.n_steps
        self.sigmas = get_sigmas_karras(self.sigma_min, self.sigma_max, self.rho, self.n_steps)

        # build diffusion
        timesteps = 1000
        sampling_timesteps = cfg.MODEL.ConsistencyDet.SAMPLE_STEP
        self.objective = 'pred_x0'
        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = 1.
        self.self_condition = False

        self.scale = 2
        self.box_renewal = True
        self.use_ensemble = True

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        self.register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # Build Dynamic Head.
        self.distillation = False

        self.head_minus = DynamicHead(cfg=cfg, roi_input_shape=self.backbone.output_shape(), distillation=self.distillation)


        if self.distillation:
            self.head = DynamicHead(cfg=cfg, roi_input_shape=self.backbone.output_shape(),distillation=self.distillation)
            #self.head.load_state_dict(self.head.state_dict(), strict=True)  # load parameters from head
            for param_minus in self.head.parameters():
            #param_minus.data.copy_(param.data)  # initialize
                param_minus.requires_grad = False  # not update by gradient

        # Loss parameters:
        class_weight = cfg.MODEL.ConsistencyDet.CLASS_WEIGHT
        giou_weight = cfg.MODEL.ConsistencyDet.GIOU_WEIGHT
        l1_weight = cfg.MODEL.ConsistencyDet.L1_WEIGHT
        #consistency_weight = 0.05
        no_object_weight = cfg.MODEL.ConsistencyDet.NO_OBJECT_WEIGHT
        self.deep_supervision = cfg.MODEL.ConsistencyDet.DEEP_SUPERVISION
        self.use_focal = cfg.MODEL.ConsistencyDet.USE_FOCAL
        self.use_fed_loss = cfg.MODEL.ConsistencyDet.USE_FED_LOSS
        self.use_nms = cfg.MODEL.ConsistencyDet.USE_NMS

        # Build Criterion.
        matcher = HungarianMatcherDynamicK(
            cfg=cfg, cost_class=class_weight, cost_bbox=l1_weight, cost_giou=giou_weight, use_focal=self.use_focal
        )

        weight_dict = {"loss_ce": class_weight, "loss_bbox": l1_weight, "loss_giou": giou_weight}
        if self.deep_supervision:
            aux_weight_dict = {}
            for i in range(self.num_heads - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "boxes"]

        self.criterion = SetCriterionDynamicK(
            cfg=cfg, num_classes=self.num_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=no_object_weight,
            losses=losses, use_focal=self.use_focal, )

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    # boxes_test cxcy01
    def model_consistency_predictions(self, backbone_feats, images_whwh, boxes_test, sig, time_cond):

        outputs_class, outputs_coord = self.consistency_function(self.head_minus, backbone_feats, boxes_test, sig, time_cond,
                                                                 images_whwh)

        x_start = outputs_coord[-1]  # (batch, num_proposals, 4) predict boxes: absolute coordinates (x1, y1, x2, y2)
        # x_start xyxy01
        x_start = x_start / images_whwh[:, None, :]
        x_start = box_xyxy_to_cxcywh(x_start)
        # x_start cxcy[-self.scale, self.scale]
        x_start = (x_start * 2 - 1.)
        # x_start cxcy[-self.scale, self.scale]。
        x_start = torch.clamp(x_start, min=-1, max=1)

        return boxes_test, x_start, outputs_class, outputs_coord

    @torch.no_grad()
    def consistency_sample(self, batched_inputs, images, images_whwh, backbone_feats, boxes_test,
                                 do_postprocess=True):

        B, N, _ = boxes_test.shape
        s_in = boxes_test.new_ones(B)
        # time = s_in * sigma
        ensemble_score, ensemble_label, ensemble_coord = [], [], []
        for step in range(self.n_steps):
            sigma = self.sigmas[step]

            # sigma_two = self.sigmas[step+1]
            time = s_in * sigma
            time = time.to(self.device)
            time_cond = torch.full((B,), step, device=self.device, dtype=torch.long)

            x_boxes, x_start, pred_cls, pred_boxes = self.model_consistency_predictions(backbone_feats, images_whwh,
                                                                                        boxes_test, time, time_cond)
            #x_boxes.to(self.device)
            #x_start.to(self.device)
            # num_remain = 1
            if self.box_renewal:  # filter
                score_per_image, box_per_image = pred_cls[-1][0], pred_boxes[-1][0]
                threshold = 0.9
                score_per_image = torch.sigmoid(score_per_image)
                value, _ = torch.max(score_per_image, -1, keepdim=False)
                keep_idx = value > threshold
                # keep_idx.to(self.device)
                num_remain = torch.sum(keep_idx)

                # pred_noise = pred_noise[:, keep_idx, :]
                x_boxes = x_boxes.to(self.device)

                x_start = x_start[:, keep_idx, :]
                x_boxes = x_boxes[:, keep_idx, :]
                # pred_cls[-1]=pred_cls[-1][:, keep_idx, :]
                # pred_boxes[-1]=pred_boxes[-1][:, keep_idx, :]
            # t_n1_min, t_n1_max = self.sigma_min, self.sigma_max

            d = (x_boxes - x_start) / append_dims(sigma, x_boxes.ndim).to(self.device)
            dt = self.sigmas[step + 1] - sigma
            # x_start_xyxys
            # x_start_xyxys = (x_start_xyxy01 * 2. - 1.) * self.scale
            # boxes_test_xyxy
            boxes_test = x_boxes + d * dt
            # print(x_boxes_xyxy01)
            # print(boxes_test)
            # boxes_test_cxcy
            # boxes_test = box_xyxy_to_cxcywh(boxes_test)

            if self.box_renewal:  # filter
                # replenish with randn boxes
                boxes_test = torch.cat((boxes_test,
                                        torch.randn(1, self.num_proposals - num_remain, 4, device=boxes_test.device) *
                                        self.sigmas[step + 1]), dim=1)
            if self.use_ensemble and self.n_steps > 1:
                box_pred_per_image, scores_per_image, labels_per_image = self.inference(pred_cls[-1], pred_boxes[-1],
                                                                                        images.image_sizes)
                ensemble_score.append(scores_per_image)
                ensemble_label.append(labels_per_image)
                ensemble_coord.append(box_pred_per_image)

        if self.use_ensemble and self.n_steps > 1:
            box_pred_per_image = torch.cat(ensemble_coord, dim=0)
            scores_per_image = torch.cat(ensemble_score, dim=0)
            labels_per_image = torch.cat(ensemble_label, dim=0)
            if self.use_nms:
                keep = batched_nms(box_pred_per_image, scores_per_image, labels_per_image, 0.6)
                box_pred_per_image = box_pred_per_image[keep]
                scores_per_image = scores_per_image[keep]
                labels_per_image = labels_per_image[keep]

            result = Instances(images.image_sizes[0])
            result.pred_boxes = Boxes(box_pred_per_image)
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results = [result]
        else:
            output = {'pred_logits': pred_cls[-1], 'pred_boxes': pred_boxes[-1]}
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            results = self.inference(box_cls, box_pred, images.image_sizes)

        if do_postprocess:
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

    # consistency model add noise
    def q_sample_consis(self, x_start, t_n1, noise=None):
        # q = 0.5
        if noise is None:
            noise = torch.randn_like(x_start)
        return x_start + noise * t_n1

    def get_scalings(self, sigma):
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        return c_skip, c_out, c_in

    def get_scalings_for_boundary_condition(self, sigma):
        c_skip = self.sigma_data ** 2 / (
                (sigma - self.sigma_min) ** 2 + self.sigma_data ** 2
        )
        c_out = (
                (sigma - self.sigma_min)
                * self.sigma_data
                / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        )
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        return c_skip, c_out * 2, c_in / 2

    def consistency_function(self, model, features, boxes, t, time_cond, images_whwh):
        if self.distillation:
            c_skip, c_out, c_in = [
                append_dims(x, boxes.ndim) for x in self.get_scalings(t)
            ]
        else:
            c_skip, c_out, c_in = [
                append_dims(x, boxes.ndim)
                for x in self.get_scalings_for_boundary_condition(t)
            ]

        c_in = c_in.to(self.device)
        boxes = boxes.to(self.device)
        bboxes = c_in * boxes
        bboxes = (bboxes + 1) / 2.
        bboxes = box_cxcywh_to_xyxy(bboxes)
        bboxes = bboxes * images_whwh[:, None, :]
        # 此处应该接受当前采样时间步
        outputs_class, outputs_coord = model(features, bboxes, time_cond, None)
        outputs_coord = outputs_coord / images_whwh[:, None, :]
        denoised_boxes = torch.clamp(outputs_coord, min=0, max=1)
        denoised_boxes = denoised_boxes * images_whwh[:, None, :]

        return outputs_class, denoised_boxes

    def forward(self, batched_inputs, do_postprocess=True):
        images, images_whwh = self.preprocess_image(batched_inputs)
        if isinstance(images, (list, torch.Tensor)):
            images = nested_tensor_from_tensor_list(images)

        # Feature Extraction.
        src = self.backbone(images.tensor)
        features = list()
        for f in self.in_features:
            feature = src[f]
            features.append(feature)

        # Prepare Proposals.
        if not self.training:
            shape = (images_whwh.shape[0], self.num_proposals, 4)
            boxes_test = torch.randn(*shape) * self.sigma_max
            results = self.consistency_sample(batched_inputs, images, images_whwh, features, boxes_test)
            return results
        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            tar, targets, boxes_n1, noises, t_n1, t_n, time_cond_n1, time_cond_n = self.prepare_targets_two(gt_instances)  # get x_(tn+1): x_boxes
            if self.distillation:
                targets, x_boxes, noises, t = self.prepare_targets(gt_instances)
                t = t.squeeze(-1)
                x_boxes = x_boxes * images_whwh[:, None, :]
            t_n1 = t_n1.squeeze(-1)
            t_n = t_n.squeeze(-1)
            time_cond_n1 = time_cond_n1.squeeze(-1)
            time_cond_n = time_cond_n.squeeze(-1)

            def solver(samples, t, next_t, x0):
                x = samples
                denoiser = x0
                dims = x0.ndim
                d = (x - denoiser) / append_dims(t, dims)
                samples = x + d * append_dims(next_t - t, dims)
                return samples

            def solver_dis(samples, t, next_t, x0, t2, x_boxes2):
                x = samples
                _, outputs_coord = self.head(features, x_boxes2, t2, None)
                denoiser = outputs_coord[-1]
                dims = x0.ndim
                d = (x - denoiser) / append_dims(t, dims)
                samples = x + d * append_dims(next_t - t, dims)
                return samples

            if self.distillation:
                boxes_n = solver_dis(boxes_n1, t_n1, t_n, tar, t, x_boxes)
            else:
                boxes_n = solver(boxes_n1, t_n1, t_n, tar)

            outputs_class, outputs_coord = self.consistency_function(self.head_minus, features, boxes_n1, t_n1,
                                                                     time_cond_n1, images_whwh)
            outputs_class_minus, outputs_coord_minus = self.consistency_function(self.head_minus, features, boxes_n, t_n,
                                                                                 time_cond_n, images_whwh)

        output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        output_minus = {'pred_logits': outputs_class_minus[-1], 'pred_boxes': outputs_coord_minus[-1]}

        if self.deep_supervision:
            output['aux_outputs'] = [{'pred_logits': a, 'pred_boxes': b}
                                     for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
            output_minus['aux_outputs'] = [{'pred_logits': a, 'pred_boxes': b}
                                           for a, b in zip(outputs_class_minus[:-1], outputs_coord_minus[:-1])]

        loss_dict1 = self.criterion(output, targets)  # self.criterion(output, output_minus, targets)
        loss_dict2 = self.criterion(output_minus, targets)  # self.criterion(output, output_minus, targets)
        # add consistency loss for pred_boxes and pred_boxes_minus
        for k in loss_dict1.keys():
            loss_dict1[k] = loss_dict1[k]+loss_dict2[k]


        weight_dict = self.criterion.weight_dict
        # weight_dict["loss_consistency"] = snrs

        for k in loss_dict1.keys():
            if k in weight_dict:
                loss_dict1[k] *= weight_dict[k]
        return loss_dict1

    def prepare_diffusion_repeat(self, gt_boxes):
        """
        :param gt_boxes: (cx, cy, w, h), normalized
        :param num_proposals:
        """
        t = torch.randint(0, self.num_timesteps, (1,), device=self.device).long()
        noise = torch.randn(self.num_proposals, 4, device=self.device)

        num_gt = gt_boxes.shape[0]
        if not num_gt:  # generate fake gt boxes if empty gt boxes
            gt_boxes = torch.as_tensor([[0.5, 0.5, 1., 1.]], dtype=torch.float, device=self.device)
            num_gt = 1

        num_repeat = self.num_proposals // num_gt  # number of repeat except the last gt box in one image
        repeat_tensor = [num_repeat] * (num_gt - self.num_proposals % num_gt) + [num_repeat + 1] * (
                self.num_proposals % num_gt)
        assert sum(repeat_tensor) == self.num_proposals
        random.shuffle(repeat_tensor)
        repeat_tensor = torch.tensor(repeat_tensor, device=self.device)

        gt_boxes = (gt_boxes * 2. - 1.) * self.scale
        x_start = torch.repeat_interleave(gt_boxes, repeat_tensor, dim=0)

        # noise sample
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        x = torch.clamp(x, min=-1 * self.scale, max=self.scale)
        x = ((x / self.scale) + 1) / 2.

        diff_boxes = box_cxcywh_to_xyxy(x)

        return diff_boxes, noise, t

    def prepare_diffusion_concat_two(self, gt_boxes):
        """
        :param gt_boxes: (cx, cy, w, h), normalized
        :param num_proposals:
        """
        num_scales = self.n_steps

        # generate noise
        noise = torch.randn(self.num_proposals, 4, device=self.device)

        num_gt = gt_boxes.shape[0]
        if not num_gt:  # generate fake gt boxes if empty gt boxes
            gt_boxes = torch.as_tensor([[0.5, 0.5, 1., 1.]], dtype=torch.float, device=self.device)
            # gt_boxes = box_cxcywh_to_xyxy(gt_boxes)
            num_gt = 1

        if num_gt < self.num_proposals:
            box_placeholder = torch.randn(self.num_proposals - num_gt, 4,
                                          device=self.device) / 6. + 0.5  # 3sigma = 1/2 --> sigma: 1/6
            box_placeholder[:, 2:] = torch.clip(box_placeholder[:, 2:], min=1e-4)
            # box_placeholder = box_cxcywh_to_xyxy(box_placeholder)
            x_start = torch.cat((gt_boxes, box_placeholder), dim=0)
        elif num_gt > self.num_proposals:
            select_mask = [True] * self.num_proposals + [False] * (num_gt - self.num_proposals)
            random.shuffle(select_mask)
            x_start = gt_boxes[select_mask]
        else:
            x_start = gt_boxes

        indices = torch.randint(0, num_scales - 1, (1,), device=self.device)

        # t_n+1
        t_n1 = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
                self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t_n1 = (t_n1 ** self.rho).to(self.device)

        # t_n
        t_n = self.sigma_max ** (1 / self.rho) + (indices + 1) / (num_scales - 1) * (
                self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t_n = (t_n ** self.rho).to(self.device)

        x_start = x_start * 2. - 1.
        x_t_n1 = self.q_sample_consis(x_start=x_start, t_n1=t_n1, noise=noise)

        # XYXY01,xyxy01,01正态分布,
        return x_start, x_t_n1, noise, t_n1, t_n, indices, indices + 1


    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def prepare_diffusion_concat(self, gt_boxes):
        """
        :param gt_boxes: (cx, cy, w, h), normalized
        :param num_proposals:
        """
        t = torch.randint(0, self.num_timesteps, (1,), device=self.device).long()
        noise = torch.randn(self.num_proposals, 4, device=self.device)

        num_gt = gt_boxes.shape[0]
        if not num_gt:  # generate fake gt boxes if empty gt boxes
            gt_boxes = torch.as_tensor([[0.5, 0.5, 1., 1.]], dtype=torch.float, device=self.device)
            num_gt = 1

        if num_gt < self.num_proposals:
            box_placeholder = torch.randn(self.num_proposals - num_gt, 4,
                                          device=self.device) / 6. + 0.5  # 3sigma = 1/2 --> sigma: 1/6
            box_placeholder[:, 2:] = torch.clip(box_placeholder[:, 2:], min=1e-4)
            x_start = torch.cat((gt_boxes, box_placeholder), dim=0)
        elif num_gt > self.num_proposals:
            select_mask = [True] * self.num_proposals + [False] * (num_gt - self.num_proposals)
            random.shuffle(select_mask)
            x_start = gt_boxes[select_mask]
        else:
            x_start = gt_boxes

        x_start = (x_start * 2. - 1.) * self.scale

        # noise sample
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        x = torch.clamp(x, min=-1 * self.scale, max=self.scale)
        x = ((x / self.scale) + 1) / 2.

        diff_boxes = box_cxcywh_to_xyxy(x)

        return diff_boxes, noise, t

    def prepare_targets(self, targets):
        new_targets = []
        diffused_boxes = []
        noises = []
        ts = []
        for targets_per_image in targets:
            target = {}
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            d_boxes, d_noise, d_t = self.prepare_diffusion_concat(gt_boxes)
            diffused_boxes.append(d_boxes)
            noises.append(d_noise)
            ts.append(d_t)
            target["labels"] = gt_classes.to(self.device)
            target["boxes"] = gt_boxes.to(self.device)
            target["boxes_xyxy"] = targets_per_image.gt_boxes.tensor.to(self.device)
            target["image_size_xyxy"] = image_size_xyxy.to(self.device)
            image_size_xyxy_tgt = image_size_xyxy.unsqueeze(0).repeat(len(gt_boxes), 1)
            target["image_size_xyxy_tgt"] = image_size_xyxy_tgt.to(self.device)
            target["area"] = targets_per_image.gt_boxes.area().to(self.device)
            new_targets.append(target)

        return new_targets, torch.stack(diffused_boxes), torch.stack(noises), torch.stack(ts)

    def prepare_targets_two(self, targets):
        new_targets = []
        diffused_boxes_n1 = []
        diffused_boxes_n = []
        noises = []
        ts1 = []
        ts = []
        ts_n1 = []
        ts_n = []
        tar = []
        for targets_per_image in targets:
            target = {}
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            # 此处不采用归一化
            # gt_boxes = targets_per_image.gt_boxes.tensor
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            # XYXY01,xyxy01,01正态分布,self.sigma_min, self.sigma_max
            x_start, d_boxes_n, d_noise, d_tn1, d_tn, time_cond_n1, time_cond_n = self.prepare_diffusion_concat_two(
                gt_boxes)

            tar.append(x_start)
            diffused_boxes_n.append(d_boxes_n)
            noises.append(d_noise)
            ts1.append(d_tn1)
            ts.append(d_tn)
            ts_n1.append(time_cond_n1)
            ts_n.append(time_cond_n)

            target["labels"] = gt_classes.to(self.device)
            target["boxes"] = gt_boxes.to(self.device)
            target["boxes_xyxy"] = targets_per_image.gt_boxes.tensor.to(self.device)
            target["image_size_xyxy"] = image_size_xyxy.to(self.device)
            image_size_xyxy_tgt = image_size_xyxy.unsqueeze(0).repeat(len(gt_boxes), 1)
            target["image_size_xyxy_tgt"] = image_size_xyxy_tgt.to(self.device)
            target["area"] = targets_per_image.gt_boxes.area().to(self.device)
            # tar.append(target["boxes_xyxy"])
            new_targets.append(target)
        # XYXY01，XYXY255，XYXY01，XYXY01，01，01
        return torch.stack(tar), new_targets, torch.stack(diffused_boxes_n), torch.stack(
            noises), torch.stack(ts1), torch.stack(ts), torch.stack(ts_n1), torch.stack(ts_n)

    def inference(self, box_cls, box_pred, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_proposals, K).
                The tensor predicts the classification probability for each proposal.
            box_pred (Tensor): tensors of shape (batch_size, num_proposals, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every proposal
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []

        if self.use_focal or self.use_fed_loss:
            scores = torch.sigmoid(box_cls)
            labels = torch.arange(self.num_classes, device=self.device). \
                unsqueeze(0).repeat(self.num_proposals, 1).flatten(0, 1)

            for i, (scores_per_image, box_pred_per_image, image_size) in enumerate(zip(
                    scores, box_pred, image_sizes
            )):
                result = Instances(image_size)
                scores_per_image, topk_indices = scores_per_image.flatten(0, 1).topk(self.num_proposals, sorted=False)
                labels_per_image = labels[topk_indices]
                box_pred_per_image = box_pred_per_image.view(-1, 1, 4).repeat(1, self.num_classes, 1).view(-1, 4)
                box_pred_per_image = box_pred_per_image[topk_indices]

                if self.use_ensemble and self.n_steps > 1:
                    return box_pred_per_image, scores_per_image, labels_per_image

                if self.use_nms:
                    keep = batched_nms(box_pred_per_image, scores_per_image, labels_per_image, 0.5)
                    box_pred_per_image = box_pred_per_image[keep]
                    scores_per_image = scores_per_image[keep]
                    labels_per_image = labels_per_image[keep]

                result.pred_boxes = Boxes(box_pred_per_image)
                result.scores = scores_per_image
                result.pred_classes = labels_per_image
                results.append(result)

        else:
            # For each box we assign the best class or the second best if the best on is `no_object`.
            scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)

            for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(zip(
                    scores, labels, box_pred, image_sizes
            )):
                if self.use_ensemble and self.n_steps > 1:
                    return box_pred_per_image, scores_per_image, labels_per_image

                if self.use_nms:
                    keep = batched_nms(box_pred_per_image, scores_per_image, labels_per_image, 0.5)
                    box_pred_per_image = box_pred_per_image[keep]
                    scores_per_image = scores_per_image[keep]
                    labels_per_image = labels_per_image[keep]
                result = Instances(image_size)
                result.pred_boxes = Boxes(box_pred_per_image)
                result.scores = scores_per_image
                result.pred_classes = labels_per_image
                results.append(result)

        return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images, self.size_divisibility)

        images_whwh = list()
        for bi in batched_inputs:
            h, w = bi["image"].shape[-2:]
            images_whwh.append(torch.tensor([w, h, w, h], dtype=torch.float32, device=self.device))
        images_whwh = torch.stack(images_whwh)

        return images, images_whwh