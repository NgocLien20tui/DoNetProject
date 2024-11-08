# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import ShapeSpec
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry

from ..backbone.resnet import BottleneckBlock, make_stage
from ..box_regression import Box2BoxTransform
from ..matcher import Matcher
from ..poolers import ROIPooler
from ..proposal_generator.proposal_utils import add_ground_truth_to_proposals
from ..sampling import subsample_labels
from .box_head import build_box_head
from .fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs
from .keypoint_head import build_keypoint_head, keypoint_rcnn_inference, keypoint_rcnn_loss
from .mask_head import build_mask_head

# From Amodal model ORCNN

# from .mask_visible_head import build_visible_mask_head
# from .mask_invisible_head import build_invisible_mask_head
# from .mask_amodal_head import build_amodal_mask_head

# from .mask_double_branch_overlapping_head import build_double_branch_overlapping_mask_head
# from .mask_double_branch_whole_head import build_double_branch_whole_mask_head

from .mask_triple_branch_overlapping_head import build_triple_branch_overlapping_mask_head
from .mask_triple_branch_nonoverlapping_head import build_triple_branch_nonoverlapping_mask_head
from .mask_triple_branch_whole_head import build_triple_branch_whole_mask_head

# from .mask_head_from_SSP import build_mask_head_from_SSP,mask_rcnn_inference, amodal_mask_rcnn_inference,\
#     mask_rcnn_loss, amodal_mask_rcnn_loss, mask_fm_loss, classes_choose
# from .recon_net import build_reconstruction_head, mask_recon_loss, mask_recon_inference
from .recls_head import build_recls_head, mask_recls_filter_loss, mask_recls_margin_loss, mask_recls_adaptive_loss
from detectron2.layers import cat, Conv2d
import os
import matplotlib.pyplot as plt

ROI_HEADS_REGISTRY = Registry("ROI_HEADS")
ROI_HEADS_REGISTRY.__doc__ = """
Registry for ROI heads in a generalized R-CNN model.
ROIHeads take feature maps and region proposals, and
perform per-region computation.

The registered object will be called with `obj(cfg, input_shape)`.
The call is expected to return an :class:`ROIHeads`.
"""

logger = logging.getLogger(__name__)


def build_roi_heads(cfg, input_shape):
    """
    Build ROIHeads defined by `cfg.MODEL.ROI_HEADS.NAME`.
    """
    name = cfg.MODEL.ROI_HEADS.NAME
    return ROI_HEADS_REGISTRY.get(name)(cfg, input_shape)


def select_foreground_proposals(
        proposals: List[Instances], bg_label: int
) -> Tuple[List[Instances], List[torch.Tensor]]:
    """
    Given a list of N Instances (for N images), each containing a `gt_classes` field,
    return a list of Instances that contain only instances with `gt_classes != -1 &&
    gt_classes != bg_label`.

    Args:
        proposals (list[Instances]): A list of N Instances, where N is the number of
            images in the batch.
        bg_label: label index of background class.

    Returns:
        list[Instances]: N Instances, each contains only the selected foreground instances.
        list[Tensor]: N boolean vector, correspond to the selection mask of
            each Instances object. True for selected instances.
    """
    assert isinstance(proposals, (list, tuple))
    assert isinstance(proposals[0], Instances)
    assert proposals[0].has("gt_classes")
    fg_proposals = []
    fg_selection_masks = []
    for proposals_per_image in proposals:
        gt_classes = proposals_per_image.gt_classes
        fg_selection_mask = (gt_classes != -1) & (gt_classes != bg_label)
        fg_idxs = fg_selection_mask.nonzero().squeeze(1)
        fg_proposals.append(proposals_per_image[fg_idxs])
        fg_selection_masks.append(fg_selection_mask)
    return fg_proposals, fg_selection_masks


def select_proposals_with_visible_keypoints(proposals: List[Instances]) -> List[Instances]:
    """
    Args:
        proposals (list[Instances]): a list of N Instances, where N is the
            number of images.

    Returns:
        proposals: only contains proposals with at least one visible keypoint.

    Note that this is still slightly different from Detectron.
    In Detectron, proposals for training keypoint head are re-sampled from
    all the proposals with IOU>threshold & >=1 visible keypoint.

    Here, the proposals are first sampled from all proposals with
    IOU>threshold, then proposals with no visible keypoint are filtered out.
    This strategy seems to make no difference on Detectron and is easier to implement.
    """
    ret = []
    all_num_fg = []
    for proposals_per_image in proposals:
        # If empty/unannotated image (hard negatives), skip filtering for train
        if len(proposals_per_image) == 0:
            ret.append(proposals_per_image)
            continue
        gt_keypoints = proposals_per_image.gt_keypoints.tensor
        # #fg x K x 3
        vis_mask = gt_keypoints[:, :, 2] >= 1
        xs, ys = gt_keypoints[:, :, 0], gt_keypoints[:, :, 1]
        proposal_boxes = proposals_per_image.proposal_boxes.tensor.unsqueeze(
            dim=1)  # #fg x 1 x 4
        kp_in_box = (
                (xs >= proposal_boxes[:, :, 0])
                & (xs <= proposal_boxes[:, :, 2])
                & (ys >= proposal_boxes[:, :, 1])
                & (ys <= proposal_boxes[:, :, 3])
        )
        selection = (kp_in_box & vis_mask).any(dim=1)
        selection_idxs = torch.nonzero(selection).squeeze(1)
        all_num_fg.append(selection_idxs.numel())
        ret.append(proposals_per_image[selection_idxs])

    storage = get_event_storage()
    storage.put_scalar("keypoint_head/num_fg_samples", np.mean(all_num_fg))
    return ret


class ROIHeads(torch.nn.Module):
    """
    ROIHeads perform all per-region computation in an R-CNN.

    It contains logic of cropping the regions, extract per-region features,
    and make per-region predictions.

    It can have many variants, implemented as subclasses of this class.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super(ROIHeads, self).__init__()

        # fmt: off
        self.batch_size_per_image = cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE
        self.positive_sample_fraction = cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
        self.test_score_thresh = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
        self.test_nms_thresh = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
        self.test_detections_per_img = cfg.TEST.DETECTIONS_PER_IMAGE
        self.in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.proposal_append_gt = cfg.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT
        self.feature_strides = {k: v.stride for k, v in input_shape.items()}
        self.feature_channels = {k: v.channels for k, v in input_shape.items()}
        self.cls_agnostic_bbox_reg = cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG
        self.smooth_l1_beta = cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA
        # fmt: on

        # Matcher to assign box proposals to gt boxes
        self.proposal_matcher = Matcher(
            cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS,
            cfg.MODEL.ROI_HEADS.IOU_LABELS,
            allow_low_quality_matches=False,
        )

        # Box2BoxTransform for bounding box regression
        self.box2box_transform = Box2BoxTransform(
            weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS)

    def _sample_proposals(
            self, matched_idxs: torch.Tensor, matched_labels: torch.Tensor, gt_classes: torch.Tensor,
            filter_out_class=-1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.

        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.
            filter_out_class (int), -1: disabled, != -1: enabled, filtering all the proposals of class == filter_out_class

        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
                each sampled proposal. Each sample is labeled as either a category in
                [0, num_classes) or the background (num_classes).
        """
        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
            gt_classes[matched_labels == 0] = self.num_classes
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
            if filter_out_class != -1:  # if filter class enabled
                gt_classes[gt_classes == filter_out_class] = 2
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes

        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
            gt_classes, self.batch_size_per_image, self.positive_sample_fraction, self.num_classes
        )

        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        return sampled_idxs, gt_classes[sampled_idxs]

    @torch.no_grad()
    def label_and_sample_proposals(
            self, proposals: List[Instances], targets: List[Instances], filter_out_class=-1
    ) -> List[Instances]:
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth
        boxes, with a fraction of positives that is no larger than
        ``self.positive_sample_fraction``.

        Args:
            See :meth:`ROIHeads.forward`
            filter_out_class (int), -1: disabled, != -1: enabled, filtering all the proposals of class == filter_out_class

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:

                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)

                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """
        gt_boxes = [x.gt_boxes for x in targets]
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head, mask head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(
                match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes, filter_out_class=filter_out_class
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            # We index all the attributes of targets that start with "gt_"
            # and have not been added to proposals yet (="gt_classes").
            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(
                            trg_name, trg_value[sampled_targets])
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros(
                        (len(sampled_idxs), 4))
                )
                proposals_per_image.gt_boxes = gt_boxes

            num_bg_samples.append(
                (gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        return proposals_with_gt

    def forward(
            self,
            images: ImageList,
            features: Dict[str, torch.Tensor],
            proposals: List[Instances],
            targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        Args:
            images (ImageList):
            features (dict[str: Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            proposals (list[Instances]): length `N` list of `Instances`. The i-th
                `Instances` contains object proposals for the i-th input image,
                with fields "proposal_boxes" and "objectness_logits".
            targets (list[Instances], optional): length `N` list of `Instances`. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.  Specify `targets` during training only.
                It may have the following fields:

                - gt_boxes: the bounding box of each instance.
                - gt_classes: the label for each instance with a category ranging in [0, #class].
                - gt_masks: PolygonMasks or BitMasks, the ground-truth masks of each instance.
                - gt_keypoints: NxKx3, the groud-truth keypoints for each instance.

        Returns:
            results (list[Instances]): length `N` list of `Instances` containing the
            detected instances. Returned during inference only; may be [] during training.

            losses (dict[str->Tensor]):
            mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        raise NotImplementedError()


@ROI_HEADS_REGISTRY.register()
class Res5ROIHeads(ROIHeads):
    """
    The ROIHeads in a typical "C4" R-CNN model, where
    the box and mask head share the cropping and
    the per-region feature computation by a Res5 block.
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)

        assert len(self.in_features) == 1

        # fmt: off
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        pooler_scales = (1.0 / self.feature_strides[self.in_features[0]],)
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        self.mask_on = cfg.MODEL.MASK_ON
        # fmt: on
        assert not cfg.MODEL.KEYPOINT_ONAmodalROIHeads

        self.pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        self.res5, out_channels = self._build_res5_block(cfg)
        self.box_predictor = FastRCNNOutputLayers(
            out_channels, self.num_classes, self.cls_agnostic_bbox_reg
        )

        if self.mask_on:
            self.mask_head = build_mask_head(
                cfg,
                ShapeSpec(channels=out_channels,
                          width=pooler_resolution, height=pooler_resolution),
            )

    def _build_res5_block(self, cfg):
        # fmt: off
        stage_channel_factor = 2 ** 3  # res5 is 8x res2
        num_groups = cfg.MODEL.RESNETS.NUM_GROUPS
        width_per_group = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
        bottleneck_channels = num_groups * width_per_group * stage_channel_factor
        out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * stage_channel_factor
        stride_in_1x1 = cfg.MODEL.RESNETS.STRIDE_IN_1X1
        norm = cfg.MODEL.RESNETS.NORM
        assert not cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE[-1], \
            "Deformable conv is not yet supported in res5 head."
        # fmt: on

        blocks = make_stage(
            BottleneckBlock,
            3,
            first_stride=2,
            in_channels=out_channels // 2,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            norm=norm,
            stride_in_1x1=stride_in_1x1,
        )
        return nn.Sequential(*blocks), out_channels

    def _shared_roi_transform(self, features, boxes):
        x = self.pooler(features, boxes)
        return self.res5(x)

    def forward(self, images, features, proposals, targets=None):
        """
        See :class:`ROIHeads.forward`.
        """
        del images

        if self.training:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
        pred_class_logits, pred_proposal_deltas = self.box_predictor(
            feature_pooled)
        del feature_pooled

        outputs = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
        )

        if self.training:
            del features
            losses = outputs.losses()
            if self.mask_on:
                proposals, fg_selection_masks = select_foreground_proposals(
                    proposals, self.num_classes
                )
                # Since the ROI feature transform is shared between boxes and masks,
                # we don't need to recompute features. The mask loss is only defined
                # on foreground proposals, so we need to select out the foreground
                # features.
                mask_features = box_features[torch.cat(
                    fg_selection_masks, dim=0)]
                del box_features
                losses.update(self.mask_head(mask_features, proposals))
            return [], losses
        else:
            pred_instances, _ = outputs.inference(
                self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img
            )
            pred_instances = self.forward_with_given_boxes(
                features, pred_instances)
            return pred_instances, {}

    def forward_with_given_boxes(self, features, instances):
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (Instances):
                the same `Instances` object, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has(
            "pred_boxes") and instances[0].has("pred_classes")

        if self.mask_on:
            features = [features[f] for f in self.in_features]
            x = self._shared_roi_transform(
                features, [x.pred_boxes for x in instances])
            return self.mask_head(x, instances)
        else:
            return instances


@ROI_HEADS_REGISTRY.register()
class StandardROIHeads(ROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    The cropped rois go to separate branches (boxes and masks) directly.
    This way, it is easier to make separate abstractions for different branches.

    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    def __init__(self, cfg, input_shape):
        super(StandardROIHeads, self).__init__(cfg, input_shape)
        self._init_box_head(cfg)
        self._init_mask_head(cfg)
        self._init_keypoint_head(cfg)
        self.for_nuclei = False
        if hasattr(cfg.MODEL.ROI_HEADS, 'FOR_NUCLEI'):
            self.for_nuclei = True

    def _init_box_head(self, cfg):
        # fmt: off
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / self.feature_strides[k] for k in self.in_features)
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        self.train_on_pred_boxes = cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [self.feature_channels[f] for f in self.in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        self.box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        self.box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels,
                           height=pooler_resolution, width=pooler_resolution)
        )
        self.box_predictor = FastRCNNOutputLayers(
            self.box_head.output_size, self.num_classes, self.cls_agnostic_bbox_reg
        )

    def _init_mask_head(self, cfg):
        # fmt: off
        self.mask_on = cfg.MODEL.MASK_ON
        if not self.mask_on:
            return
        pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / self.feature_strides[k] for k in self.in_features)
        sampling_ratio = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [self.feature_channels[f] for f in self.in_features][0]

        self.mask_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        self.mask_head = build_mask_head(
            cfg, ShapeSpec(channels=in_channels,
                           width=pooler_resolution, height=pooler_resolution)
        )

    def _init_keypoint_head(self, cfg):
        # fmt: off
        self.keypoint_on = cfg.MODEL.KEYPOINT_ON
        if not self.keypoint_on:
            return
        pooler_resolution = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / self.feature_strides[k] for k in self.in_features)  # noqa
        sampling_ratio = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_TYPE
        self.normalize_loss_by_visible_keypoints = cfg.MODEL.ROI_KEYPOINT_HEAD.NORMALIZE_LOSS_BY_VISIBLE_KEYPOINTS  # noqa
        self.keypoint_loss_weight = cfg.MODEL.ROI_KEYPOINT_HEAD.LOSS_WEIGHT
        # fmt: on

        in_channels = [self.feature_channels[f] for f in self.in_features][0]

        self.keypoint_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        self.keypoint_head = build_keypoint_head(
            cfg, ShapeSpec(channels=in_channels,
                           width=pooler_resolution, height=pooler_resolution)
        )

    def forward(
            self,
            images: ImageList,
            features: Dict[str, torch.Tensor],
            proposals: List[Instances],
            targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        if self.training:
            assert targets
            if self.for_nuclei:
                proposals = self.label_and_sample_proposals(proposals, targets, filter_out_class=0)
            else:
                proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        features_list = [features[f] for f in self.in_features]

        if self.training:
            losses = self._forward_box(features_list, proposals)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            losses.update(self._forward_mask(features_list, proposals))
            losses.update(self._forward_keypoint(features_list, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features_list, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(
                features, pred_instances)
            return pred_instances, {}

    def forward_with_given_boxes(
            self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> List[Instances]:
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        This is useful for downstream tasks where a box is known, but need to obtain
        other attributes (outputs of other heads).
        Test-time augmentation also uses this.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (list[Instances]):
                the same `Instances` objects, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has(
            "pred_boxes") and instances[0].has("pred_classes")
        features_list = [features[f] for f in self.in_features]
        instances = self._forward_mask(features_list, instances)
        instances = self._forward_keypoint(features_list, instances)
        return instances

    def _forward_box(
            self, features: List[torch.Tensor], proposals: List[Instances]
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (list[Tensor]): #level input features for box prediction
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        box_features = self.box_pooler(
            features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        pred_class_logits, pred_proposal_deltas = self.box_predictor(
            box_features)
        del box_features

        outputs = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
        )
        if self.training:
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = outputs.predict_boxes_for_gt_classes()
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(
                            pred_boxes_per_image)
            return outputs.losses()
        else:
            pred_instances, _ = outputs.inference(
                self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img
            )
            return pred_instances

    def _forward_mask(
            self, features: List[torch.Tensor], instances: List[Instances]
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the mask prediction branch.

        Args:
            features (list[Tensor]): #level input features for mask prediction
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """
        if not self.mask_on:
            return {} if self.training else instances

        if self.training:
            # The loss is only defined on positive proposals.
            proposals, _ = select_foreground_proposals(
                instances, self.num_classes)
            proposal_boxes = [x.proposal_boxes for x in proposals]
            mask_features = self.mask_pooler(features, proposal_boxes)
            return self.mask_head(mask_features, proposals)
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            mask_features = self.mask_pooler(features, pred_boxes)
            return self.mask_head(mask_features, instances)

    def _forward_keypoint(
            self, features: List[torch.Tensor], instances: List[Instances]
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the keypoint prediction branch.

        Args:
            features (list[Tensor]): #level input features for keypoint prediction
            instances (list[Instances]): the per-image instances to train/predict keypoints.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_keypoints" and return it.
        """
        if not self.keypoint_on:
            return {} if self.training else instances

        num_images = len(instances)

        if self.training:
            # The loss is defined on positive proposals with at >=1 visible keypoints.
            proposals, _ = select_foreground_proposals(
                instances, self.num_classes)
            proposals = select_proposals_with_visible_keypoints(proposals)
            proposal_boxes = [x.proposal_boxes for x in proposals]

            keypoint_features = self.keypoint_pooler(features, proposal_boxes)
            keypoint_logits = self.keypoint_head(keypoint_features)

            normalizer = (
                    num_images
                    * self.batch_size_per_image
                    * self.positive_sample_fraction
                    * keypoint_logits.shape[1]
            )
            loss = keypoint_rcnn_loss(
                keypoint_logits,
                proposals,
                normalizer=None if self.normalize_loss_by_visible_keypoints else normalizer,
            )
            return {"loss_keypoint": loss * self.keypoint_loss_weight}
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            keypoint_features = self.keypoint_pooler(features, pred_boxes)
            keypoint_logits = self.keypoint_head(keypoint_features)
            keypoint_rcnn_inference(keypoint_logits, instances)
            return instances


#
# @ROI_HEADS_REGISTRY.register()
# class AmodalROIHeads(ROIHeads):
#     """
#     A Standard ROIHeads which contains additional heads for the prediction of amodal masks (amodal mask head)
# and the occlusion mask (occlusion mask head).
#     """
#
#     def __init__(self, cfg, input_shape):
#         super(AmodalROIHeads, self).__init__(cfg, input_shape)
#         self._init_box_head(cfg)
#         self._init_amodal_mask_head(cfg)
#         self._init_visible_mask_head(cfg)
#         self._init_invisible_mask_head(cfg)
#
#     def _init_box_head(self, cfg):
#         # fmt: off
#         pooler_resolution        = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
#         pooler_scales            = tuple(1.0 / self.feature_strides[k] for k in self.in_features)
#         sampling_ratio           = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
#         pooler_type              = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
#         self.train_on_pred_boxes = cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES
#         # fmt: on
#
#         # If StandardROIHeads is applied on multiple feature maps (as in FPN),
#         # then we share the same predictors and therefore the channel counts must be the same
#         in_channels = [self.feature_channels[f] for f in self.in_features]
#         # Check all channel counts are equal
#         assert len(set(in_channels)) == 1, in_channels
#         in_channels = in_channels[0]
#
#         self.box_pooler = ROIPooler(
#             output_size=pooler_resolution,
#             scales=pooler_scales,
#             sampling_ratio=sampling_ratio,
#             pooler_type=pooler_type,
#         )
#         # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
#         # They are used together so the "box predictor" layers should be part of the "box head".
#         # New subclasses of ROIHeads do not need "box predictor"s.
#         self.box_head = build_box_head(
#             cfg, ShapeSpec(channels=in_channels,
#                            height=pooler_resolution, width=pooler_resolution)
#         )
#         self.box_predictor = FastRCNNOutputLayers(
#             self.box_head.output_size, self.num_classes, self.cls_agnostic_bbox_reg
#         )
#
#     def _init_amodal_mask_head(self, cfg):
#         # fmt: off
#         self.mask_on           = cfg.MODEL.MASK_ON
#         if not self.mask_on:
#             return
#
#         pooler_resolution = cfg.MODEL.ROI_AMODAL_MASK_HEAD.POOLER_RESOLUTION
#         pooler_scales     = tuple(1.0 / self.feature_strides[k] for k in self.in_features)
#         sampling_ratio    = cfg.MODEL.ROI_AMODAL_MASK_HEAD.POOLER_SAMPLING_RATIO
#         pooler_type       = cfg.MODEL.ROI_AMODAL_MASK_HEAD.POOLER_TYPE
#         # fmt: on
#         in_channels = [self.feature_channels[f] for f in self.in_features][0]
#         self.mask_pooler = ROIPooler(
#             output_size=pooler_resolution,
#             scales=pooler_scales,
#             sampling_ratio=sampling_ratio,
#             pooler_type=pooler_type,
#         )
#         self.amodal_mask_head = build_amodal_mask_head(
#             cfg, ShapeSpec(channels=in_channels,
#                            width=pooler_resolution, height=pooler_resolution)
#         )
#
#     def _init_visible_mask_head(self, cfg):
#         # fmt: off
#         self.mask_on           = cfg.MODEL.MASK_ON
#         if not self.mask_on:
#             return
#         pooler_resolution = cfg.MODEL.ROI_VISIBLE_MASK_HEAD.POOLER_RESOLUTION
#         # fmt: on
#         in_channels = [self.feature_channels[f] for f in self.in_features][0]
#         self.visible_mask_head = build_visible_mask_head(
#             cfg, ShapeSpec(channels=in_channels,
#                            width=pooler_resolution, height=pooler_resolution)
#         )
#
#     def _init_invisible_mask_head(self, cfg):
#         # fmt: off
#         self.mask_on           = cfg.MODEL.MASK_ON
#         if not self.mask_on:
#             return
#         self.invisible_mask_head = build_invisible_mask_head(cfg)
#
#     def _forward_amodal_mask(self, features: List[torch.Tensor], instances: List[Instances]):
#         """
#         Forward logic of the mask prediction branch.
#
#         Args:
#             features (list[Tensor]): #level input features for mask prediction
#             instances (list[Instances]): the per-image instances to train/predict masks.
#                 In training, they can be the proposals.
#                 In inference, they can be the predicted boxes.
#
#         Returns:
#             In training, a dict of losses and pred_mask_logits
#             In inference, update `instances` with new fields "pred_masks" and return it and pred_mask_logits
#         """
#         if not self.mask_on:
#             return {} if self.training else instances
#         if self.training:
#             # The loss is only defined on positive proposals.
#             proposals, _ = select_foreground_proposals(instances, self.num_classes)
#             proposal_boxes = [x.proposal_boxes for x in proposals]
#             mask_features = self.mask_pooler(features, proposal_boxes)
#             return self.amodal_mask_head(mask_features, proposals)
#         else:
#             pred_boxes = [x.pred_boxes for x in instances]
#             mask_features = self.mask_pooler(features, pred_boxes)
#             return self.amodal_mask_head(mask_features, instances)
#
#     def _forward_visible_mask(self, features: List[torch.Tensor], instances: List[Instances]):
#         """
#         Forward logic of the mask prediction branch.
#
#         Args:
#             features (list[Tensor]): #level input features for mask prediction
#             instances (list[Instances]): the per-image instances to train/predict masks.
#                 In training, they can be the proposals.
#                 In inference, they can be the predicted boxes.
#
#         Returns:
#             In training, a dict of losses and pred_mask_logits
#             In inference, update `instances` with new fields "pred_masks" and return it and pred_mask_logits
#         """
#         if not self.mask_on:
#             return {} if self.training else instances
#         if self.training:
#             # The loss is only defined on positive proposals.
#             proposals, _ = select_foreground_proposals(instances, self.num_classes)
#             proposal_boxes = [x.proposal_boxes for x in proposals]
#             mask_features = self.mask_pooler(features, proposal_boxes)
#             return self.visible_mask_head(mask_features, proposals)
#         else:
#             pred_boxes = [x.pred_boxes for x in instances]
#             mask_features = self.mask_pooler(features, pred_boxes)
#             return self.visible_mask_head(mask_features, instances)
#
#     def _forward_invisible_mask(self,pred_amodal_mask_logits,pred_visible_mask_logits,instances):
#         if not self.mask_on:
#             return {} if self.training else instances
#
#         if self.training:
#             # The loss is only defined on positive proposals.
#             proposals, _ = select_foreground_proposals(instances, self.num_classes)
#             pred_invisible_mask_logtis = pred_amodal_mask_logits - F.relu(pred_visible_mask_logits)
#             return self.invisible_mask_head(pred_invisible_mask_logtis,proposals)
#         else:
#             pred_invisible_mask_logtis = pred_amodal_mask_logits - F.relu(pred_visible_mask_logits)
#             return self.invisible_mask_head(pred_invisible_mask_logtis, instances)
#
#
#     def forward(
#         self,
#         images: ImageList,
#         features: Dict[str, torch.Tensor],
#         proposals: List[Instances],
#         targets: Optional[List[Instances]] = None,
#     ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
#         """
#         See :class:`ROIHeads.forward`.
#         """
#         del images
#         if self.training:
#             assert targets
#             proposals = self.label_and_sample_proposals(proposals, targets)
#         del targets
#
#         features_list = [features[f] for f in self.in_features]
#
#         if self.training:
#             losses = self._forward_box(features_list, proposals)
#             # Usually the original proposals used by the box head are used by the mask, keypoint
#             # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
#             # predicted by the box head.
#             amodal_mask_loss,amodal_mask_logits = self._forward_amodal_mask(features_list, proposals)
#             losses.update(amodal_mask_loss )
#             visible_mask_loss,visible_mask_logits = self._forward_visible_mask(features_list, proposals)
#             losses.update(visible_mask_loss)
#             losses.update(self._forward_invisible_mask(amodal_mask_logits, visible_mask_logits,proposals))
#             return proposals, losses
#         else:
#             pred_instances = self._forward_box(features_list, proposals)
#             # During inference cascaded prediction is used: the mask and keypoints heads are only
#             # applied to the top scoring box detections.
#             pred_instances = self.forward_with_given_boxes(features, pred_instances)
#             return pred_instances, {}
#
#     def _forward_box(
#         self, features: List[torch.Tensor], proposals: List[Instances]
#     ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
#         """
#         Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
#             the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.
#
#         Args:
#             features (list[Tensor]): #level input features for box prediction
#             proposals (list[Instances]): the per-image object proposals with
#                 their matching ground truth.
#                 Each has fields "proposal_boxes", and "objectness_logits",
#                 "gt_classes", "gt_boxes".
#
#         Returns:
#             In training, a dict of losses.
#             In inference, a list of `Instances`, the predicted instances.
#         """
#         box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
#         box_features = self.box_head(box_features)
#         pred_class_logits, pred_proposal_deltas = self.box_predictor(box_features)
#         del box_features
#
#         outputs = FastRCNNOutputs(
#             self.box2box_transform,
#             pred_class_logits,
#             pred_proposal_deltas,
#             proposals,
#             self.smooth_l1_beta,
#         )
#         if self.training:
#             if self.train_on_pred_boxes:
#                 with torch.no_grad():
#                     pred_boxes = outputs.predict_boxes_for_gt_classes()
#                     for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
#                         proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
#             return outputs.losses()
#         else:
#             pred_instances, _ = outputs.inference(
#                 self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img
#             )
#             return pred_instances
#
#     def forward_with_given_boxes(
#         self, features: Dict[str, torch.Tensor], instances: List[Instances]
#     ) -> List[Instances]:
#         """
#         Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.
#
#         This is useful for downstream tasks where a box is known, but need to obtain
#         other attributes (outputs of other heads).
#         Test-time augmentation also uses this.
#
#         Args:
#             features: same as in `forward()`
#             instances (list[Instances]): instances to predict other outputs. Expect the keys
#                 "pred_boxes" and "pred_classes" to exist.
#
#         Returns:
#             instances (list[Instances]):
#                 the same `Instances` objects, with extra
#                 fields such as `pred_masks` or `pred_keypoints`. and 'pred_visible_masks'
#         """
#
#         assert not self.training
#         assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")
#         features_list = [features[f] for f in self.in_features]
#         instances,amodal_mask_logits = self._forward_amodal_mask(features_list, instances)
#         instances,visible_mask_logits = self._forward_visible_mask(features_list, instances)
#         instances = self._forward_invisible_mask(amodal_mask_logits, visible_mask_logits, instances)
#         return instances
#
# @ROI_HEADS_REGISTRY.register()
# class DoubleBranchROIHeads(ROIHeads):
#     """
#     A Standard ROIHeads which contains additional heads for the prediction of amodal masks (amodal mask head)
# and the occlusion mask (occlusion mask head).
#     """
#
#     def __init__(self, cfg, input_shape):
#         super(DoubleBranchROIHeads, self).__init__(cfg, input_shape)
#         self._init_box_head(cfg)
#         self._init_double_branch_whole_mask_head(cfg)
#         self._init_double_branch_overlapping_mask_head(cfg)
#
#         self.heads_attention = False
#         if hasattr(cfg.MODEL.ROI_DOUBLE_BRANCH_WHOLE_MASK_HEAD, 'HEADS_ATTENTION'):
#             self.heads_attention = cfg.MODEL.ROI_DOUBLE_BRANCH_WHOLE_MASK_HEAD.HEADS_ATTENTION
#         self.pure_branch = False
#         if hasattr(cfg.MODEL.ROI_DOUBLE_BRANCH_WHOLE_MASK_HEAD, 'PURE_BRANCH'):
#             self.pure_branch = cfg.MODEL.ROI_DOUBLE_BRANCH_WHOLE_MASK_HEAD.PURE_BRANCH
#         self.rpn_attention_db = False
#         if hasattr(cfg.MODEL,'RPN_ATTENTION'):
#             self.rpn_attention_db = cfg.MODEL.RPN_ATTENTION
#
#     def _init_box_head(self, cfg):
#         # fmt: off
#         pooler_resolution        = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
#         pooler_scales            = tuple(1.0 / self.feature_strides[k] for k in self.in_features)
#         sampling_ratio           = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
#         pooler_type              = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
#         self.train_on_pred_boxes = cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES
#         # fmt: on
#
#         # If StandardROIHeads is applied on multiple feature maps (as in FPN),
#         # then we share the same predictors and therefore the channel counts must be the same
#         in_channels = [self.feature_channels[f] for f in self.in_features]
#         # Check all channel counts are equal
#         assert len(set(in_channels)) == 1, in_channels
#         in_channels = in_channels[0]
#
#         self.box_pooler = ROIPooler(
#             output_size=pooler_resolution,
#             scales=pooler_scales,
#             sampling_ratio=sampling_ratio,
#             pooler_type=pooler_type,
#         )
#         # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
#         # They are used together so the "box predictor" layers should be part of the "box head".
#         # New subclasses of ROIHeads do not need "box predictor"s.
#         self.box_head = build_box_head(
#             cfg, ShapeSpec(channels=in_channels,
#                            height=pooler_resolution, width=pooler_resolution)
#         )
#         self.box_predictor = FastRCNNOutputLayers(
#             self.box_head.output_size, self.num_classes, self.cls_agnostic_bbox_reg
#         )
#
#     def _init_double_branch_whole_mask_head(self, cfg):
#         # fmt: off
#         self.mask_on           = cfg.MODEL.MASK_ON
#         if not self.mask_on:
#             return
#
#         pooler_resolution = cfg.MODEL.ROI_DOUBLE_BRANCH_WHOLE_MASK_HEAD.POOLER_RESOLUTION
#         pooler_scales     = tuple(1.0 / self.feature_strides[k] for k in self.in_features)
#         sampling_ratio    = cfg.MODEL.ROI_DOUBLE_BRANCH_WHOLE_MASK_HEAD.POOLER_SAMPLING_RATIO
#         pooler_type       = cfg.MODEL.ROI_DOUBLE_BRANCH_WHOLE_MASK_HEAD.POOLER_TYPE
#         # fmt: on
#         in_channels = [self.feature_channels[f] for f in self.in_features][0]
#         self.mask_pooler = ROIPooler(
#             output_size=pooler_resolution,
#             scales=pooler_scales,
#             sampling_ratio=sampling_ratio,
#             pooler_type=pooler_type,
#         )
#         self.double_branch_whole_mask_head = build_double_branch_whole_mask_head(
#             cfg, ShapeSpec(channels=in_channels,
#                            width=pooler_resolution, height=pooler_resolution)
#         )
#
#     def _init_double_branch_overlapping_mask_head(self, cfg):
#         # fmt: off
#         self.mask_on           = cfg.MODEL.MASK_ON
#         if not self.mask_on:
#             return
#         pooler_resolution = cfg.MODEL.ROI_DOUBLE_BRANCH_OVERLAPPING_MASK_HEAD.POOLER_RESOLUTION
#         # fmt: on
#         in_channels = [self.feature_channels[f] for f in self.in_features][0]
#         self.double_branch_overlapping_mask_head = build_double_branch_overlapping_mask_head(
#             cfg, ShapeSpec(channels=in_channels,
#                            width=pooler_resolution, height=pooler_resolution)
#         )
#
#     def _forward_double_branch_whole_mask(self, features: List[torch.Tensor], instances: List[Instances], guiding_layers = True):
#         """
#         Forward logic of the mask prediction branch.
#
#         Args:
#             features (list[Tensor]): #level input features for mask prediction
#             instances (list[Instances]): the per-image instances to train/predict masks.
#                 In training, they can be the proposals.
#                 In inference, they can be the predicted boxes.
#
#         Returns:
#             In training, a dict of losses and pred_mask_logits
#             In inference, update `instances` with new fields "pred_masks" and return it and pred_mask_logits
#         """
#         if not self.mask_on:
#             return {} if self.training else instances
#         if self.training:
#             return self.double_branch_whole_mask_head(features, instances, guiding_layers)
#         else:
#             return self.double_branch_whole_mask_head(features, instances, guiding_layers)
#
#     def _forward_double_branch_overlapping_mask(self, features: List[torch.Tensor], instances: List[Instances]):
#         """
#         Forward logic of the mask prediction branch.
#
#         Args:
#             features (list[Tensor]): #level input features for mask prediction
#             instances (list[Instances]): the per-image instances to train/predict masks.
#                 In training, they can be the proposals.
#                 In inference, they can be the predicted boxes.
#
#         Returns:
#             In training, a dict of losses and pred_mask_logits
#             In inference, update `instances` with new fields "pred_masks" and return it and pred_mask_logits
#         """
#         if not self.mask_on:
#             return {} if self.training else instances
#         if self.training:
#             return self.double_branch_overlapping_mask_head(features, instances)
#         else:
#             return self.double_branch_overlapping_mask_head(features, instances)
#
#     def _forward_box(
#         self, features: List[torch.Tensor], proposals: List[Instances]
#     ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
#         """
#         Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
#             the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.
#
#         Args:
#             features (list[Tensor]): #level input features for box prediction
#             proposals (list[Instances]): the per-image object proposals with
#                 their matching ground truth.
#                 Each has fields "proposal_boxes", and "objectness_logits",
#                 "gt_classes", "gt_boxes".
#
#         Returns:
#             In training, a dict of losses.
#             In inference, a list of `Instances`, the predicted instances.
#         """
#         box_features = self.box_pooler(
#             features, [x.proposal_boxes for x in proposals])
#         box_features = self.box_head(box_features)
#         pred_class_logits, pred_proposal_deltas = self.box_predictor(
#             box_features)
#         del box_features
#
#         outputs = FastRCNNOutputs(
#             self.box2box_transform,
#             pred_class_logits,
#             pred_proposal_deltas,
#             proposals,
#             self.smooth_l1_beta,
#         )
#         if self.training:
#             if self.train_on_pred_boxes:
#                 with torch.no_grad():
#                     pred_boxes = outputs.predict_boxes_for_gt_classes()
#                     for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
#                         proposals_per_image.proposal_boxes = Boxes(
#                             pred_boxes_per_image)
#             return outputs.losses()
#         else:
#             pred_instances, _ = outputs.inference(
#                 self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img
#             )
#             return pred_instances
#
#     def forward_with_given_boxes(
#         self, features: Dict[str, torch.Tensor], instances: List[Instances]
#     ) -> List[Instances]:
#         """
#         Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.
#
#         This is useful for downstream tasks where a box is known, but need to obtain
#         other attributes (outputs of other heads).
#         Test-time augmentation also uses this.
#
#         Args:
#             features: same as in `forward()`
#             instances (list[Instances]): instances to predict other outputs. Expect the keys
#                 "pred_boxes" and "pred_classes" to exist.
#
#         Returns:
#             instances (list[Instances]):
#                 the same `Instances` objects, with extra
#                 fields such as `pred_double_branch_whole_masks` or `pred_keypoints`. and 'pred_double_branch_overlapping_masks'
#             if attention: whole_mask_logits
#         """
#         # TODO: concate
#         assert not self.training
#         assert instances[0].has(
#             "pred_boxes") and instances[0].has("pred_classes")
#
#         # TODO TODO TODO
#         features_list = [features[f] for f in self.in_features]
#         pred_boxes = [x.pred_boxes for x in instances]
#         mask_features = self.mask_pooler(features_list, pred_boxes)
#
#         if self.heads_attention:
#             _, _, double_branch_whole_mask_attention = self._forward_double_branch_whole_mask(mask_features, instances, guiding_layers = False)
#
#             mask_features_for_over_branches = mask_features*double_branch_whole_mask_attention
#         else:
#             mask_features_for_over_branches = mask_features
#
#         instances, double_branch_overlapping_mask_logits, double_branch_overlapping_mask_inner_feature = self._forward_double_branch_overlapping_mask(
#             mask_features_for_over_branches, instances)
#
#         if self.pure_branch:
#             instances, whole_mask_logits, double_branch_whole_mask_inner_feature = self._forward_double_branch_whole_mask(
#                 mask_features, instances, guiding_layers = False)
#         else:
#             concatenated_mask_feature = torch.cat(
#                 [mask_features]+[double_branch_overlapping_mask_inner_feature], 1)
#
#             instances, whole_mask_logits, double_branch_whole_mask_inner_feature = self._forward_double_branch_whole_mask(
#                 concatenated_mask_feature, instances)
#         if self.rpn_attention_db:
#             return instances, whole_mask_logits
#         return instances
#
#     def forward(
#         self,
#         images: ImageList,
#         features: Dict[str, torch.Tensor],
#         proposals: List[Instances],
#         targets: Optional[List[Instances]] = None,
#     ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
#         """
#         See :class:`ROIHeads.forward`.
#         """
#         del images
#         if self.training:
#             assert targets
#             if self.rpn_attention_db:
#                 proposals = self.label_and_sample_proposals(proposals, targets, filter_out_class=1) # only dealing with cells
#             else:
#                 proposals = self.label_and_sample_proposals(proposals, targets)
#         del targets
#
#         features_list = [features[f] for f in self.in_features]
#
#         if self.training:
#             losses = self._forward_box(features_list, proposals)
#             # Usually the original proposals used by the box head are used by the mask, keypoint
#             # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
#             # predicted by the box head.
#             # Heads TODO:
#             # TODO: chop features from proposals √
#             selected_proposals, _ = select_foreground_proposals(
#                 proposals, self.num_classes)
#             proposal_boxes = [x.proposal_boxes for x in selected_proposals]
#             mask_features = self.mask_pooler(features_list, proposal_boxes)
#
#             # Head TODO:
#             # TODO: modify so that it can give the inner feature
#
#             if self.heads_attention:
#                 double_branch_whole_attention_mask_loss, _, double_branch_whole_mask_attention = self._forward_double_branch_whole_mask(mask_features, selected_proposals, guiding_layers = False)
#                 # double_branch_whole_attention_mask_loss = {key:value*0.5 for key, value in double_branch_whole_attention_mask_loss.items()}
#                 losses.update(double_branch_whole_attention_mask_loss)
#                 mask_features_for_over_branches = mask_features * double_branch_whole_mask_attention
#
#             else:
#                 mask_features_for_over_branches = mask_features
#
#
#             # TODO: make overlapping branch from visible branch, remember to change data input
#             double_branch_overlapping_mask_loss, double_branch_overlapping_mask_logits, double_branch_overlapping_mask_inner_feature = self._forward_double_branch_overlapping_mask(
#                 mask_features_for_over_branches, selected_proposals)
#             losses.update(double_branch_overlapping_mask_loss)
#
#             if self.pure_branch:
#                 double_branch_whole_mask_loss, double_branch_whole_mask_logits, double_branch_whole_mask_inner_feature = self._forward_double_branch_whole_mask(
#                     mask_features, selected_proposals, guiding_layers=False)
#                 losses.update(double_branch_whole_mask_loss)
#             else:
#                 # TODO: concate √
#                 concatenated_mask_feature = torch.cat(
#                     [mask_features]+[double_branch_overlapping_mask_inner_feature], 1)
#                 # TODO: make Conv layers in whole branch
#                 double_branch_whole_mask_loss, double_branch_whole_mask_logits, double_branch_whole_mask_inner_feature = self._forward_double_branch_whole_mask(
#                     concatenated_mask_feature, selected_proposals)
#                 losses.update(double_branch_whole_mask_loss)
#
#             if self.rpn_attention_db:
#                 return proposals, losses, double_branch_whole_mask_logits, selected_proposals
#             return proposals, losses
#         else: # inference
#             pred_instances = self._forward_box(features_list, proposals)
#             # During inference cascaded prediction is used: the mask and keypoints heads are only
#             # applied to the top scoring box detections.
#             if self.rpn_attention_db:
#                 pred_instances, double_branch_whole_mask_logits = self.forward_with_given_boxes(
#                     features, pred_instances)
#                 return pred_instances, {}, double_branch_whole_mask_logits
#
#             pred_instances = self.forward_with_given_boxes(
#                 features, pred_instances)
#             return pred_instances, {}


@ROI_HEADS_REGISTRY.register()
class TripleBranchROIHeads(ROIHeads):
    """
    A Standard ROIHeads which contains additional heads for the prediction of amodal masks (amodal mask head)
and the occlusion mask (occlusion mask head).
    """

    def __init__(self, cfg, input_shape):
        super(TripleBranchROIHeads, self).__init__(cfg, input_shape)
        self.vis_period = cfg.VIS_PERIOD
        self.consis_loss = cfg.MODEL.ROI_TRIPLE_BRANCH_WHOLE_MASK_HEAD.CONSIS_LOSS
        self.consis_loss_mode = cfg.MODEL.ROI_TRIPLE_BRANCH_WHOLE_MASK_HEAD.CONSIS_LOSS_MODE

        self._init_box_head(cfg)
        self._init_triple_branch_whole_mask_head(cfg)
        self._init_triple_branch_overlapping_mask_head(cfg)
        self._init_triple_branch_nonoverlapping_mask_head(cfg)

        self.for_nuclei = False
        if hasattr(cfg.MODEL.ROI_HEADS, 'FOR_NUCLEI'):
            self.for_nuclei = True

        self.unlabeled = False
        self.semi_supervised = False
        if hasattr(cfg.MODEL, 'SEMI'):
            self.semi_supervised = cfg.MODEL.SEMI
        self.pure_branch = False
        if hasattr(cfg.MODEL.ROI_TRIPLE_BRANCH_WHOLE_MASK_HEAD, 'PURE_BRANCH'):
            self.pure_branch = cfg.MODEL.ROI_TRIPLE_BRANCH_WHOLE_MASK_HEAD.PURE_BRANCH
        self.rpn_attention_tb = False
        if hasattr(cfg.MODEL, 'RPN_ATTENTION'):
            self.rpn_attention_tb = cfg.MODEL.RPN_ATTENTION

        self.branch_guidance = False
        if hasattr(cfg.MODEL.ROI_TRIPLE_BRANCH_WHOLE_MASK_HEAD, 'BRANCH_GUIDANCE'):
            self.branch_guidance = cfg.MODEL.ROI_TRIPLE_BRANCH_WHOLE_MASK_HEAD.BRANCH_GUIDANCE
        self.heads_attention = False
        if hasattr(cfg.MODEL.ROI_TRIPLE_BRANCH_WHOLE_MASK_HEAD, 'HEADS_ATTENTION'):
            self.heads_attention = cfg.MODEL.ROI_TRIPLE_BRANCH_WHOLE_MASK_HEAD.HEADS_ATTENTION
        self.refinement = True
        if hasattr(cfg.MODEL.ROI_TRIPLE_BRANCH_WHOLE_MASK_HEAD, 'REFINEMENT'):
            self.refinement = cfg.MODEL.ROI_TRIPLE_BRANCH_WHOLE_MASK_HEAD.REFINEMENT
            if not self.refinement:
                assert self.branch_guidance, "refinement cannot be off without branch_guidance"

        if self.branch_guidance:
            self.branch_guidance_conv_overlap = []
            n_features = 2
            conv_dims = cfg.MODEL.ROI_TRIPLE_BRANCH_WHOLE_MASK_HEAD.CONV_DIM
            n_features = 2
            self.branch_guidance_conv_overlap.append(
                Conv2d(n_features * conv_dims, n_features * conv_dims, 3, 1, 1, activation=F.relu))
            self.branch_guidance_conv_overlap.append(
                Conv2d(n_features * conv_dims, n_features * conv_dims, 3, 1, 1, activation=F.relu))
            self.branch_guidance_conv_overlap.append(
                Conv2d(n_features * conv_dims, conv_dims, 3, 1, 1, activation=F.relu))
            for i, layer in enumerate(self.branch_guidance_conv_overlap):
                self.add_module(
                    "branch_guidance_conv_overlap_layer{}".format(i), layer)
            self.branch_guidance_conv_non_overlap = []
            n_features = 2
            conv_dims = cfg.MODEL.ROI_TRIPLE_BRANCH_WHOLE_MASK_HEAD.CONV_DIM
            n_features = 2
            self.branch_guidance_conv_non_overlap.append(
                Conv2d(n_features * conv_dims, n_features * conv_dims, 3, 1, 1, activation=F.relu))
            self.branch_guidance_conv_non_overlap.append(
                Conv2d(n_features * conv_dims, n_features * conv_dims, 3, 1, 1, activation=F.relu))
            self.branch_guidance_conv_non_overlap.append(
                Conv2d(n_features * conv_dims, conv_dims, 3, 1, 1, activation=F.relu))
            for i, layer in enumerate(self.branch_guidance_conv_non_overlap):
                self.add_module(
                    "branch_guidance_conv_non_overlap_layer{}".format(i), layer)

    def _init_box_head(self, cfg):
        # fmt: off
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / self.feature_strides[k] for k in self.in_features)
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        self.train_on_pred_boxes = cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [self.feature_channels[f] for f in self.in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        self.box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        self.box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels,
                           height=pooler_resolution, width=pooler_resolution)
        )
        self.box_predictor = FastRCNNOutputLayers(
            self.box_head.output_size, self.num_classes, self.cls_agnostic_bbox_reg
        )

    def _init_triple_branch_whole_mask_head(self, cfg):
        # fmt: off
        self.mask_on = cfg.MODEL.MASK_ON
        if not self.mask_on:
            return

        pooler_resolution = cfg.MODEL.ROI_TRIPLE_BRANCH_WHOLE_MASK_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / self.feature_strides[k] for k in self.in_features)
        sampling_ratio = cfg.MODEL.ROI_TRIPLE_BRANCH_WHOLE_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_TRIPLE_BRANCH_WHOLE_MASK_HEAD.POOLER_TYPE
        # fmt: on
        in_channels = [self.feature_channels[f] for f in self.in_features][0]
        self.mask_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        self.triple_branch_whole_mask_head = build_triple_branch_whole_mask_head(
            cfg, ShapeSpec(channels=in_channels,
                           width=pooler_resolution, height=pooler_resolution)
        )

    def _init_triple_branch_overlapping_mask_head(self, cfg):
        # fmt: off
        self.mask_on = cfg.MODEL.MASK_ON
        if not self.mask_on:
            return
        pooler_resolution = cfg.MODEL.ROI_TRIPLE_BRANCH_OVERLAPPING_MASK_HEAD.POOLER_RESOLUTION
        # fmt: on
        in_channels = [self.feature_channels[f] for f in self.in_features][0]
        self.triple_branch_overlapping_mask_head = build_triple_branch_overlapping_mask_head(
            cfg, ShapeSpec(channels=in_channels,
                           width=pooler_resolution, height=pooler_resolution)
        )

    def _init_triple_branch_nonoverlapping_mask_head(self, cfg):
        # fmt: off
        self.mask_on = cfg.MODEL.MASK_ON
        if not self.mask_on:
            return
        pooler_resolution = cfg.MODEL.ROI_TRIPLE_BRANCH_NONOVERLAPPING_MASK_HEAD.POOLER_RESOLUTION
        # fmt: on
        in_channels = [self.feature_channels[f] for f in self.in_features][0]
        self.triple_branch_nonoverlapping_mask_head = build_triple_branch_nonoverlapping_mask_head(
            cfg, ShapeSpec(channels=in_channels,
                           width=pooler_resolution, height=pooler_resolution)
        )

    def _forward_triple_branch_whole_mask(self, features: List[torch.Tensor], instances: List[Instances],
                                          guiding_layers=True):
        """
        Forward logic of the mask prediction branch.

        Args:
            features (list[Tensor]): #level input features for mask prediction
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses and pred_mask_logits
            In inference, update `instances` with new fields "pred_masks" and return it and pred_mask_logits
        """
        if not self.mask_on:
            return {} if self.training else instances
        if self.training:
            return self.triple_branch_whole_mask_head(features, instances, guiding_layers)
        else:
            return self.triple_branch_whole_mask_head(features, instances, guiding_layers)

    def _forward_triple_branch_overlapping_mask(self, features: List[torch.Tensor], instances: List[Instances]):
        """
        Forward logic of the mask prediction branch.

        Args:
            features (list[Tensor]): #level input features for mask prediction
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses and pred_mask_logits
            In inference, update `instances` with new fields "pred_masks" and return it and pred_mask_logits
        """
        if not self.mask_on:
            return {} if self.training else instances
        if self.training:
            return self.triple_branch_overlapping_mask_head(features, instances)
        else:
            return self.triple_branch_overlapping_mask_head(features, instances)

    def _forward_triple_branch_nonoverlapping_mask(self, features: List[torch.Tensor], instances: List[Instances]):
        """
        Forward logic of the mask prediction branch.

        Args:
            features (list[Tensor]): #level input features for mask prediction
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses and pred_mask_logits
            In inference, update `instances` with new fields "pred_masks" and return it and pred_mask_logits
        """
        if not self.mask_on:
            return {} if self.training else instances
        if self.training:
            return self.triple_branch_nonoverlapping_mask_head(features, instances)
        else:
            return self.triple_branch_nonoverlapping_mask_head(features, instances)

    def _forward_box(
            self, features: List[torch.Tensor], proposals: List[Instances]
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (list[Tensor]): #level input features for box prediction
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        box_features = self.box_pooler(
            features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        pred_class_logits, pred_proposal_deltas = self.box_predictor(
            box_features)
        del box_features

        outputs = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
        )
        if self.training:
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = outputs.predict_boxes_for_gt_classes()
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(
                            pred_boxes_per_image)
            return outputs.losses()
        else:
            pred_instances, _ = outputs.inference(
                self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img
            )
            return pred_instances

    def forward_with_given_boxes(
            self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> List[Instances]:
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        This is useful for downstream tasks where a box is known, but need to obtain
        other attributes (outputs of other heads).
        Test-time augmentation also uses this.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (list[Instances]):
                the same `Instances` objects, with extra
                fields such as `pred_triple_branch_whole_masks` or `pred_keypoints`. and 'pred_triple_branch_overlapping_masks'
        """
        # TODO: concate
        assert not self.training
        assert instances[0].has(
            "pred_boxes") and instances[0].has("pred_classes")

        # TODO TODO TODO
        features_list = [features[f] for f in self.in_features]
        pred_boxes = [x.pred_boxes for x in instances]
        mask_features = self.mask_pooler(features_list, pred_boxes)

        if self.branch_guidance:
            instances, _, triple_branch_whole_mask_inner_feature_for_branch_guidance = self._forward_triple_branch_whole_mask(
                mask_features, instances, guiding_layers=False)
            mask_features_for_over_nonover_branches = torch.cat(
                [mask_features] + [triple_branch_whole_mask_inner_feature_for_branch_guidance], 1)
            for layer in self.branch_guidance_conv_overlap:
                mask_features_for_over_branches = layer(mask_features_for_over_nonover_branches)
            for layer in self.branch_guidance_conv_non_overlap:
                mask_features_for_nonover_branches = layer(mask_features_for_over_nonover_branches)
        elif self.heads_attention:
            _, _, triple_branch_whole_mask_attention = self._forward_triple_branch_whole_mask(mask_features, instances,
                                                                                              guiding_layers=False)
            mask_features_for_over_nonover_branches = mask_features * triple_branch_whole_mask_attention
        else:
            mask_features_for_over_nonover_branches = mask_features

        if self.branch_guidance:
            instances, triple_branch_overlapping_mask_logits, triple_branch_overlapping_mask_inner_feature = self._forward_triple_branch_overlapping_mask(
                mask_features_for_over_branches, instances)

            instances, triple_branch_nonoverlapping_mask_logits, triple_branch_nonoverlapping_mask_inner_feature = self._forward_triple_branch_nonoverlapping_mask(
                mask_features_for_nonover_branches, instances)
        else:
            instances, triple_branch_overlapping_mask_logits, triple_branch_overlapping_mask_inner_feature = self._forward_triple_branch_overlapping_mask(
                mask_features_for_over_nonover_branches, instances)

            instances, triple_branch_nonoverlapping_mask_logits, triple_branch_nonoverlapping_mask_inner_feature = self._forward_triple_branch_nonoverlapping_mask(
                mask_features_for_over_nonover_branches, instances)

        if self.refinement:
            if self.pure_branch:
                instances, whole_mask_logits, triple_branch_whole_mask_inner_feature = self._forward_triple_branch_whole_mask(
                    mask_features, instances, guiding_layers=False)
            else:
                concatenated_mask_feature = torch.cat(
                    [mask_features] + [triple_branch_overlapping_mask_inner_feature] + [
                        triple_branch_nonoverlapping_mask_inner_feature], 1)

                instances, whole_mask_logits, triple_branch_whole_mask_inner_feature = self._forward_triple_branch_whole_mask(
                    concatenated_mask_feature, instances)
        if self.rpn_attention_tb:
            return instances, whole_mask_logits
        return instances

    def cal_triple_branch_consistency_loss(self, pred_mask_logits_from_main_branch, pred_mask_logits_from_addition,
                                           vis_period=0):

        storage = get_event_storage()
        if vis_period > 0 and storage.iter % vis_period == 0:
            from_main_branch = (
                                   pred_mask_logits_from_main_branch.sigmoid())[:, 0]
            from_addition = (pred_mask_logits_from_addition.sigmoid())[:, 0]
            vis_masks = torch.cat([from_main_branch, from_addition], axis=2)
            name = "Left: triple_branch_whole from_main_branch;   Right: triple_branch_whole from_addition"
            for idx, vis_mask in enumerate(vis_masks):
                vis_mask = torch.stack([vis_mask] * 3, axis=0)
                storage.put_image(name + f" ({idx})", vis_mask)

        consistency_loss = F.binary_cross_entropy(
            input=pred_mask_logits_from_addition, target=pred_mask_logits_from_main_branch, reduction="mean")
        return consistency_loss

    def linear_normalization(self, x, eps=0.001):
        return (x - x.min()) / max(x.max() - x.min(), eps)

    def forward(
            self,
            images: ImageList,
            features: Dict[str, torch.Tensor],
            proposals: List[Instances],
            targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        if self.training:
            self.unlabeled = (len(targets[0]) == 0)
        if self.training and not self.unlabeled:
            assert targets
            if self.for_nuclei:
                proposals = self.label_and_sample_proposals(proposals, targets, filter_out_class=0)
            elif self.rpn_attention_tb:
                proposals = self.label_and_sample_proposals(proposals, targets,
                                                            filter_out_class=1)  # only dealing with cells
            else:
                proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        features_list = [features[f] for f in self.in_features]

        features_valid = [torch.isfinite(features[f]).all() for f in self.in_features]

        if features_valid.count(False) != 0:
            print("Non finite features appears!")

        if self.training:
            if self.unlabeled:
                losses = {}
            else:
                losses = self._forward_box(features_list, proposals)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            # Heads TODO:
            # TODO: chop features from proposals √
            if self.unlabeled:
                selected_proposals = proposals
            else:
                selected_proposals, _ = select_foreground_proposals(
                    proposals, self.num_classes)
            proposal_boxes = [x.proposal_boxes for x in selected_proposals]
            mask_features = self.mask_pooler(features_list, proposal_boxes)

            # Head TODO:
            # TODO: modify so that it can give the inner feature

            if self.branch_guidance:
                triple_branch_whole_attention_mask_loss, triple_branch_whole_attention_mask_logits, triple_branch_whole_mask_inner_feature_for_branch_guidance = self._forward_triple_branch_whole_mask(
                    mask_features, selected_proposals, guiding_layers=False)
                if not self.unlabeled:
                    triple_branch_whole_attention_mask_loss = {key: value * 0.75 for key, value in
                                                               triple_branch_whole_attention_mask_loss.items()}
                    losses.update(triple_branch_whole_attention_mask_loss)
                mask_features_for_over_nonover_branches = torch.cat(
                    [mask_features] + [triple_branch_whole_mask_inner_feature_for_branch_guidance], 1)
                for layer in self.branch_guidance_conv_overlap:
                    mask_features_for_over_branches = layer(mask_features_for_over_nonover_branches)
                for layer in self.branch_guidance_conv_non_overlap:
                    mask_features_for_nonover_branches = layer(mask_features_for_over_nonover_branches)
            elif self.heads_attention:
                triple_branch_whole_attention_mask_loss, _, triple_branch_whole_mask_attention = self._forward_triple_branch_whole_mask(
                    mask_features, selected_proposals, guiding_layers=False)
                # triple_branch_whole_attention_mask_loss = {key:value*0.5 for key, value in triple_branch_whole_attention_mask_loss.items()}
                if self.unlabeled:
                    del triple_branch_whole_attention_mask_loss
                else:
                    losses.update(triple_branch_whole_attention_mask_loss)

                mask_features_for_over_nonover_branches = mask_features * triple_branch_whole_mask_attention
            else:
                mask_features_for_over_nonover_branches = mask_features

            if self.branch_guidance:
                # TODO: make overlapping branch from visible branch, remember to change data input
                triple_branch_overlapping_mask_loss, triple_branch_overlapping_mask_logits, triple_branch_overlapping_mask_inner_feature = self._forward_triple_branch_overlapping_mask(
                    mask_features_for_over_branches, selected_proposals)
                if self.unlabeled:
                    del triple_branch_overlapping_mask_loss
                else:
                    losses.update(triple_branch_overlapping_mask_loss)

                # TODO: make nonoverlapping branch from visible branch, remember to change data input
                triple_branch_nonoverlapping_mask_loss, triple_branch_nonoverlapping_mask_logits, triple_branch_nonoverlapping_mask_inner_feature = self._forward_triple_branch_nonoverlapping_mask(
                    mask_features_for_nonover_branches, selected_proposals)
                if self.unlabeled:
                    del triple_branch_nonoverlapping_mask_loss
                else:
                    losses.update(triple_branch_nonoverlapping_mask_loss)
            else:
                # TODO: make overlapping branch from visible branch, remember to change data input
                triple_branch_overlapping_mask_loss, triple_branch_overlapping_mask_logits, triple_branch_overlapping_mask_inner_feature = self._forward_triple_branch_overlapping_mask(
                    mask_features_for_over_nonover_branches, selected_proposals)
                if self.unlabeled:
                    del triple_branch_overlapping_mask_loss
                else:
                    losses.update(triple_branch_overlapping_mask_loss)

                # TODO: make nonoverlapping branch from visible branch, remember to change data input
                triple_branch_nonoverlapping_mask_loss, triple_branch_nonoverlapping_mask_logits, triple_branch_nonoverlapping_mask_inner_feature = self._forward_triple_branch_nonoverlapping_mask(
                    mask_features_for_over_nonover_branches, selected_proposals)
                if self.unlabeled:
                    del triple_branch_nonoverlapping_mask_loss
                else:
                    losses.update(triple_branch_nonoverlapping_mask_loss)

            if self.refinement:
                if self.pure_branch:
                    triple_branch_whole_mask_loss, triple_branch_whole_mask_logits, triple_branch_whole_mask_inner_feature = self._forward_triple_branch_whole_mask(
                        mask_features, selected_proposals, guiding_layers=False)
                    losses.update(triple_branch_whole_mask_loss)
                else:
                    # TODO: concate √
                    concatenated_mask_feature = torch.cat(
                        [mask_features] + [triple_branch_overlapping_mask_inner_feature] + [
                            triple_branch_nonoverlapping_mask_inner_feature], 1)
                    # TODO: make Conv layers in whole branch
                    triple_branch_whole_mask_loss, triple_branch_whole_mask_logits, triple_branch_whole_mask_inner_feature = self._forward_triple_branch_whole_mask(
                        concatenated_mask_feature, selected_proposals)
                if self.unlabeled:
                    del triple_branch_whole_mask_loss
                else:
                    losses.update(triple_branch_whole_mask_loss)

            if self.branch_guidance and self.unlabeled:
                losses.update({"branch_guidance_consistency_loss": 0.1 * F.binary_cross_entropy(
                    input=(triple_branch_whole_attention_mask_logits.sigmoid()).sigmoid(),
                    target=(triple_branch_whole_mask_logits.sigmoid()).sigmoid(), reduction="mean")})

            if self.consis_loss:
                if self.consis_loss_mode == "MX":  # mask xor
                    triple_branch_main_mask_loss_input = (triple_branch_whole_mask_logits.sigmoid() > 0.5).float()
                    triple_branch_addition_mask_loss_input = (
                                triple_branch_nonoverlapping_mask_logits.sigmoid() > 0.5).logical_xor(
                        triple_branch_overlapping_mask_logits.sigmoid() > 0.5).float()
                    triple_branch_consistency_loss = {
                        "loss_triple_branch_consistency": self.cal_triple_branch_consistency_loss(
                            triple_branch_main_mask_loss_input, triple_branch_addition_mask_loss_input,
                            self.vis_period) * 0.00125}

                losses.update(triple_branch_consistency_loss)
            if self.rpn_attention_tb:
                return proposals, losses, triple_branch_whole_mask_logits, selected_proposals
            return proposals, losses
        else:
            pred_instances = self._forward_box(features_list, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            if self.rpn_attention_tb:
                pred_instances, triple_branch_whole_mask_logits = self.forward_with_given_boxes(
                    features, pred_instances)
                return pred_instances, {}, triple_branch_whole_mask_logits

            pred_instances = self.forward_with_given_boxes(
                features, pred_instances)
            return pred_instances, {}


def get_pred_masks_logits_by_cls(pred_mask_logits, instances):
    cls_agnostic_mask = pred_mask_logits.size(1) == 1
    total_num_masks = pred_mask_logits.size(0)

    if isinstance(instances, list):
        mask_side_len = pred_mask_logits.size(2)
        assert pred_mask_logits.size(2) == pred_mask_logits.size(3), "Mask prediction must be square!"

        gt_classes = []
        gt_masks = []
        for instances_per_image in instances:
            if len(instances_per_image) == 0:
                continue
            if not cls_agnostic_mask:
                gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
                gt_classes.append(gt_classes_per_image)
            gt_masks_per_image = instances_per_image.gt_masks.crop_and_resize(
                instances_per_image.proposal_boxes.tensor, mask_side_len
            ).to(device=pred_mask_logits.device)
            # num_regions*28*28
            # A tensor of shape (N, M, M), N=#instances in the image; M=mask_side_len
            gt_masks.append(gt_masks_per_image)
        if len(gt_masks) == 0:
            return pred_mask_logits.sum() * 0
        gt_masks = cat(gt_masks, dim=0)

        if cls_agnostic_mask:
            pred_mask_logits = pred_mask_logits[:, 0]
        else:
            indices = torch.arange(total_num_masks)
            gt_classes = cat(gt_classes, dim=0)
            pred_mask_logits = pred_mask_logits[indices, gt_classes]

        gt_masks = gt_masks.float()

        return pred_mask_logits.unsqueeze(1), gt_masks.unsqueeze(1)
    else:
        indices = torch.arange(total_num_masks)
        gt_classes = instances

        pred_mask_logits = pred_mask_logits[indices, gt_classes]

        return pred_mask_logits.unsqueeze(1)
