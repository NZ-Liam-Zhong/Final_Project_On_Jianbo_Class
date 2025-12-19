# Some parts of this code is assisted by AI Agent
from functools import partial
from types import MappingProxyType
from typing import Any, Union

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import center_of_mass

from src.backbone import *
from src.dataset import *


def conv_gn_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, num_groups=32, bias=False):
    """Helper function to create a Conv2d -> GroupNorm -> ReLU layer."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
        nn.GroupNorm(num_groups, out_channels),
        nn.ReLU(inplace=False)
    )


class SOLOHead(nn.Module):
    NUM_FPN_LAYERS = 5
    PREDICTION_MASK_SHAPE = (200, 272)

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 256,
        seg_feat_channels: int = 256,
        stacked_convs: int = 7,
        strides: tuple[int, ...] = (8, 8, 16, 32, 32),
        num_groups: int = 32,
        scale_ranges: tuple[tuple[int, int], ...] = ((1, 96), (48, 192), (96, 384), (192, 768), (384, 2048)),
        grid_sizes: tuple[int, ...] = (40, 36, 24, 16, 12),
        mask_loss_cfg: dict[str, float] = MappingProxyType({"weight": 3}),
        cate_loss_cfg: dict[str, float] = MappingProxyType({"gamma": 2, "alpha": 0.25, "weight": 1}),
        postprocess_cfg: dict[str, Union[int, float]] = MappingProxyType({
            "cate_thresh": 0.2,
            "ins_thresh": 0.5,
            "keep_instance": 5,
        })
    ):
        super().__init__()
        self.num_classes = num_classes
        self.grid_sizes = grid_sizes
        self.cate_out_channels = self.num_classes - 1
        self.in_channels = in_channels
        self.seg_feat_channels = seg_feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.num_groups = num_groups
        self.scale_ranges = scale_ranges

        self.mask_loss_cfg = mask_loss_cfg
        self.cate_loss_cfg = cate_loss_cfg
        self.postprocess_cfg = postprocess_cfg

        # Initialize the layers
        self._init_layers()

        # Check consistency
        assert len(self.instance_out_list) == len(self.strides)

    def _init_layers(self):
        """
        This function builds network layers for category and instance branches.
        It initializes:
          - self.cate_head: nn.ModuleList of intermediate layers for category branch
          - self.ins_head: nn.ModuleList of intermediate layers for instance branch
          - self.cate_out: Output layer for category branch
          - self.ins_out_list: nn.ModuleList of output layers for instance branch, one for each FPN level
        """
        ################################################################################################################
        # TODO: Implement the classification branch. Should have a total of (3 * self.stacked_convs + 2) modules
        # Hint: Use the helper function ``conv_gn_relu``
        ################################################################################################################
        # Category branch head - 7 conv layers + 1 output layer
        category_layers = []
        for i in range(self.stacked_convs):
            category_layers.append(
                conv_gn_relu(
                    self.in_channels, 
                    self.seg_feat_channels, 
                    kernel_size=3, 
                    stride=1, 
                    padding=1, 
                    num_groups=self.num_groups, 
                    bias=False
                )
            )
        
        # Output layer for category branch
        category_layers.append(
            nn.Conv2d(
                self.seg_feat_channels, 
                self.cate_out_channels, 
                kernel_size=3, 
                stride=1, 
                padding=1, 
                bias=True
            )
        )
        category_layers.append(nn.Sigmoid())
        
        self.category_branch = nn.Sequential(*category_layers)
        ################################################################################################################

        ################################################################################################################
        # TODO: Implement the instance branch. Should have a total of (3 * self.stacked_convs) modules
        # Hint: Use the helper function ``conv_gn_relu``
        ################################################################################################################
        # Instance branch head - 7 conv layers (no output layer here, it's separate for each FPN level)
        instance_layers = []
        
        # First layer takes FPN features + 2 coordinate channels
        instance_layers.append(
            conv_gn_relu(
                self.in_channels + 2,  # +2 for coordinate channels
                self.seg_feat_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                num_groups=self.num_groups,
                bias=False
            )
        )
        
        # Remaining layers
        for i in range(self.stacked_convs - 1):
            instance_layers.append(
                conv_gn_relu(
                    self.seg_feat_channels,
                    self.seg_feat_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    num_groups=self.num_groups,
                    bias=False
                )
            )
        
        self.instance_branch = nn.Sequential(*instance_layers)
        ################################################################################################################

        ################################################################################################################
        # TODO: Implement a separate segmentation head for each layer of the FPN. Each one should have 2 modules
        ################################################################################################################
        # Instance branch output layers - one for each FPN level
        self.instance_out_list = nn.ModuleList()
        
        for grid_size in self.grid_sizes:
            out_channels = grid_size ** 2  # S^2 output channels for each grid
            layer = nn.Sequential(
                nn.Conv2d(
                    self.seg_feat_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True
                ),
                nn.Sigmoid()
            )
            self.instance_out_list.append(layer)
        ################################################################################################################

    def forward(self, fpn_feature_dict: dict[str, torch.Tensor]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Forward function processes every level in the FPN.
        Input:
        - fpn_feat_dict (dict[str, torch.Tensor]): Dictionary structure of FPN features
        Output:
        - category_prediction_list (list[torch.Tensor]): num_fpn_layers x (bsz, S, S, C)
        - instance_prediction_list (list[torch.Tensor]): num_fpn_layers x (bsz, S^2, H, W)
        """
        fpn_feature_list = [v.to(next(self.parameters()).device) for v in fpn_feature_dict.values()]
        fpn_feature_list = self.adjust_fpn_feature_strides(fpn_feature_list)  # Adjust FPN features to desired strides

        category_prediction_list, instance_prediction_list = self.multi_apply(
            self.forward_for_single_level,
            fpn_feature_list,
            [*range(len(fpn_feature_list)),],
        )

        # Check flag
        assert len(fpn_feature_list) == len(self.grid_sizes)
        assert instance_prediction_list[1].shape[1] == self.grid_sizes[1] ** 2
        assert category_prediction_list[1].shape[2] == self.grid_sizes[1]

        return category_prediction_list, instance_prediction_list

    def adjust_fpn_feature_strides(self, fpn_feature_list: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Adjust the original FPN feature maps to have strides [8, 8, 16, 32, 32].
        The sizes of the feature maps are adjusted by interpolation.
        """
        # Adjust level 0 and level 4 feature maps
        fpn_p2 = F.interpolate(fpn_feature_list[0], size=fpn_feature_list[1].shape[-2:], mode="bilinear", align_corners=False)
        fpn_p5 = F.interpolate(fpn_feature_list[4], size=fpn_feature_list[3].shape[-2:], mode="bilinear", align_corners=False)
        new_fpn = [fpn_p2, fpn_feature_list[1], fpn_feature_list[2], fpn_feature_list[3], fpn_p5]
        return new_fpn

    def forward_for_single_level(
        self,
        fpn_features: torch.Tensor,
        idx: int,
    ):
        """
        This function forwards a single level of FPN feature map through the network.
        Input:
        - fpn_features: (batch_size, fpn_channels, H_feat, W_feat)
        - idx: Index of the FPN level
        Output:
        - category_predictions: Category prediction
        - instance_predictions: Instance prediction
        """

        grid_size = self.grid_sizes[idx]
        batch_size = fpn_features.shape[0]

        ################################################################################################################
        # TODO: Compute the category predictions using the FPN features
        ################################################################################################################
        # Category branch (temporary stub for shape validation)
        # Category branch: predict heatmap then resize to (S, S)
        category_logits = self.category_branch(fpn_features)
        category_predictions = F.interpolate(
            category_logits,
            size=(grid_size, grid_size),
            mode="bilinear",
            align_corners=False,
        )
        ################################################################################################################

        ################################################################################################################
        # TODO: Compute the instance mask predictions using the FPN and coordinate features
        ################################################################################################################
        # Instance branch (temporary stub for shape validation)
        # Instance branch: concatenate coord features, conv tower, upsample x2, then level-specific 1x1 -> S^2 channels
        coord = self.generate_coordinates(
            (fpn_features.shape[2], fpn_features.shape[3]),
            fpn_features.device,
        )  # (2, H, W)
        coord = coord.unsqueeze(0).expand(batch_size, -1, -1, -1)
        ins_in = torch.cat([fpn_features, coord], dim=1)
        ins_feats = self.instance_branch(ins_in)
        ins_feats_up = F.interpolate(
            ins_feats,
            scale_factor=2.0,
            mode="bilinear",
            align_corners=False,
        )
        instance_predictions = self.instance_out_list[idx](ins_feats_up)
        ################################################################################################################

        # Check flag
        assert category_predictions.shape[1:] == (3, grid_size, grid_size)
        assert instance_predictions.shape[1:] == (grid_size ** 2, fpn_features.shape[2] * 2, fpn_features.shape[3] * 2)

        return category_predictions, instance_predictions

    def points_nms(self, category_predictions: torch.Tensor):
        """
        This function applies NMS on the heat map (category_predictions), grid level
        Input:
        - category_predictions: (batch_size, C - 1, S, S)
        Output:
        - category_predictions after NMS
        """
        hmax = F.max_pool2d(category_predictions, kernel_size=2, stride=1, padding=1)
        keep = (hmax[..., :-1, :-1] == category_predictions).float()
        return category_predictions * keep

    def generate_coordinates(self, shape, device):
        """
        Generate coordinate feature map.
        """
        x_range = torch.linspace(0, 1, shape[1], device=device)
        y_range = torch.linspace(0, 1, shape[0], device=device)
        y, x = torch.meshgrid(y_range, x_range)
        return torch.stack([x, y], dim=0)  # (2, H, W)

    def multi_apply(self, func, *args, **kwargs) -> tuple[Any, ...]:
        """
        Apply function to a list of arguments.
        """
        pfunc = partial(func, **kwargs) if kwargs else func
        map_results = map(pfunc, *args)
        return (*zip(*map_results),)

    def construct_targets(self, bbox_list, label_list, mask_list, target_sizes: list[torch.Size] = None):
        """
        Build the ground truth tensor for each batch in the training.
        Input:
        - bbox_list: List of bounding boxes for each image in the batch
        - label_list: List of labels for each image in the batch
        - mask_list: List of masks for each image in the batch
        Output:
        - instance_ground_truth_list: List of instance ground truths
        - instance_index_ground_truth_list: List of instance indices
        - category_grouund_truth_list: List of category ground truths
        """
        if target_sizes is None:
            target_sizes = [SOLOHead.PREDICTION_MASK_SHAPE] * SOLOHead.NUM_FPN_LAYERS
        target_sizes = [target_sizes] * len(mask_list)

        output = map(self.construct_target_for_single_image, bbox_list, label_list, mask_list, target_sizes)
        instance_ground_truth_list, instance_index_ground_truth_list, category_grouund_truth_list = zip(*output)

        # Check flag
        assert instance_ground_truth_list[0][1].shape == (self.grid_sizes[1] ** 2,) + SOLOHead.PREDICTION_MASK_SHAPE
        assert instance_index_ground_truth_list[0][1].shape == (self.grid_sizes[1] ** 2,)
        assert category_grouund_truth_list[0][1].shape == (self.grid_sizes[1], self.grid_sizes[1])

        return instance_ground_truth_list, instance_index_ground_truth_list, category_grouund_truth_list

    def construct_target_for_single_image(
        self,
        bounding_boxes: torch.Tensor,
        labels: torch.Tensor,
        masks: torch.Tensor,
        featmap_sizes: list[torch.Size],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """
        Process single image to generate target labels for each feature pyramid level.
        Input:
        - bounding_boxes (tensor): Shape (n_obj, 4) in x1y1x2y2 format
        - labels (tensor): Shape (n_obj,)
        - masks (tensor): Shape (n_obj, H_ori, W_ori)
        - featmap_sizes (list): Sizes of feature maps for each level
        Output:
        - instance_mask_list (list[tensor]): num_fpn_layers x (S^2, H, W)
        - instance_confidence_list (list[tensor]): num_fpn_layers x (S^2,)
        - category_list (list[tensor]): num_fpn_layers x (S, S)
        """

        # Lists to store labels for each feature map level
        instance_mask_list = []
        instance_confidence_list = []
        category_list = []

        ################################################################################################################
        # TODO: Use the information from the dataset to construct training labels for the SOLO head output
        ################################################################################################################
        device = bounding_boxes.device if len(bounding_boxes) > 0 else torch.device('cpu')
        
        # Process each FPN level
        for level_idx in range(self.NUM_FPN_LAYERS):
            grid_size = self.grid_sizes[level_idx]
            scale_range = self.scale_ranges[level_idx]
            featmap_size = featmap_sizes[level_idx] if featmap_sizes else SOLOHead.PREDICTION_MASK_SHAPE
            
            # Initialize targets for this level
            instance_masks = torch.zeros((grid_size ** 2, featmap_size[0], featmap_size[1]), 
                                       dtype=torch.float32, device=device)
            instance_confidences = torch.zeros(grid_size ** 2, dtype=torch.float32, device=device)
            # Use -1 for background so focal loss can ignore it cleanly
            category_targets = torch.full((grid_size, grid_size), -1, dtype=torch.long, device=device)
            
            if len(bounding_boxes) == 0:
                instance_mask_list.append(instance_masks)
                instance_confidence_list.append(instance_confidences)
                category_list.append(category_targets)
                continue
            
            # Calculate object scales (sqrt of area)
            w = bounding_boxes[:, 2] - bounding_boxes[:, 0]
            h = bounding_boxes[:, 3] - bounding_boxes[:, 1]
            object_scales = torch.sqrt(w * h)
            
            # Find objects that belong to this scale range
            scale_mask = (object_scales >= scale_range[0]) & (object_scales <= scale_range[1])
            
            if not scale_mask.any():
                instance_mask_list.append(instance_masks)
                instance_confidence_list.append(instance_confidences)
                category_list.append(category_targets)
                continue
            
            # Get objects for this level
            level_bboxes = bounding_boxes[scale_mask]
            level_labels = labels[scale_mask]
            level_masks = masks[scale_mask]
            
            # Calculate center points of bounding boxes
            center_x = (level_bboxes[:, 0] + level_bboxes[:, 2]) / 2
            center_y = (level_bboxes[:, 1] + level_bboxes[:, 3]) / 2
            
            # Map centers to grid coordinates
            # Assuming image size is 800x1088 (after padding)
            img_h, img_w = 800, 1088
            grid_x = (center_x / img_w * grid_size).long().clamp(0, grid_size - 1)
            grid_y = (center_y / img_h * grid_size).long().clamp(0, grid_size - 1)
            
            # Assign targets
            for obj_idx in range(len(level_bboxes)):
                gx, gy = grid_x[obj_idx], grid_y[obj_idx]
                label = level_labels[obj_idx]
                mask = level_masks[obj_idx]
                
                # Calculate grid index (row-major order)
                grid_idx = gy * grid_size + gx
                
                # Resize mask to feature map size
                # mask shape is (1, 800, 1088), need to add batch and channel dims
                if mask.dim() == 3:
                    mask = mask.squeeze(0)  # Remove channel dim if present: (800, 1088)
                
                mask_resized = torch.nn.functional.interpolate(
                    mask.unsqueeze(0).unsqueeze(0).float(),  # Add batch and channel dims: (1, 1, 800, 1088)
                    size=featmap_size,
                    mode='bilinear',
                    align_corners=False
                ).squeeze()  # Remove batch and channel dims: (200, 272)
                
                # Assign instance mask and confidence
                instance_masks[grid_idx] = mask_resized
                instance_confidences[grid_idx] = 1.0
                
                # Assign category (subtract 1 because background is not included)
                category_targets[gy, gx] = label - 1
            
            instance_mask_list.append(instance_masks)
            instance_confidence_list.append(instance_confidences)
            category_list.append(category_targets)
        ################################################################################################################

        return instance_mask_list, instance_confidence_list, category_list

    def loss(self, category_prediction_list, instance_prediction_list, instance_ground_truth_list, instance_index_ground_truth_list, category_ground_truth_list):
        """
        Compute loss for a batch of images.
        """
        instance_ground_truths = [
            torch.cat([
                ins_labels_level_img[ins_ind_labels_level_img > 0, ...]
                for ins_labels_level_img, ins_ind_labels_level_img in zip(ins_labels_level, ins_ind_labels_level)
            ], dim=0)
            for ins_labels_level, ins_ind_labels_level in zip(zip(*instance_ground_truth_list), zip(*instance_index_ground_truth_list))
        ]
        instance_predictions_filtered = [
            torch.cat([
                ins_preds_level_img[ins_ind_labels_level_img > 0, ...]
                for ins_preds_level_img, ins_ind_labels_level_img in zip(ins_preds_level, ins_ind_labels_level)
            ], dim=0)
            for ins_preds_level, ins_ind_labels_level in zip(instance_prediction_list, zip(*instance_index_ground_truth_list))
        ]

        # Category targets/preds
        category_ground_truths = torch.cat([
            torch.cat([
                cate_gts_level_img.flatten()
                for cate_gts_level_img in cate_gts_level
            ], dim=0)
            for cate_gts_level in zip(*category_ground_truth_list)
        ], dim=0).to(torch.long)
        category_predictions_flat = torch.cat([
            cate_pred_level.permute(0, 2, 3, 1).reshape(-1, self.cate_out_channels)
            for cate_pred_level in category_prediction_list
        ], dim=0)

        # Losses
        # Compute dice loss PER FPN LEVEL to avoid spatial size mismatch, then average
        dice_losses = []
        for preds_level, gts_level in zip(instance_predictions_filtered, instance_ground_truths):
            if gts_level.numel() == 0 or preds_level.numel() == 0:
                continue
            gt_level = gts_level
            if preds_level.shape[-2:] != gt_level.shape[-2:]:
                gt_level = F.interpolate(gt_level.unsqueeze(1), size=preds_level.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
            dice_losses.append(self.dice_loss(preds_level, gt_level).mean())
        dice_loss_val = torch.stack(dice_losses).mean() if dice_losses else torch.tensor(0.0, device=category_predictions_flat.device)

        focal_loss_val = self.focal_loss(category_predictions_flat, category_ground_truths).mean()
        total_loss = self.cate_loss_cfg["weight"] * focal_loss_val + self.mask_loss_cfg["weight"] * dice_loss_val

        return total_loss, focal_loss_val, dice_loss_val

    def dice_loss(
        self,
        mask_prediction: torch.Tensor,          # (..., 2H_feat, 2W_feat,)
        mask_ground_truth: torch.Tensor,        # (..., 2H_feat, 2W_feat,)
    ) -> torch.Tensor:                          # (...,)
        """
        Compute the dice loss.
        """
        ################################################################################################################
        # TODO: Compute the dice loss
        ################################################################################################################
        # Flatten spatial dims
        eps = 1e-6
        p = mask_prediction.reshape(mask_prediction.shape[0], -1)
        q = mask_ground_truth.reshape(mask_ground_truth.shape[0], -1)
        # Dice coefficient per mask
        intersection = (p * q).sum(dim=1)
        denom = (p.pow(2).sum(dim=1) + q.pow(2).sum(dim=1)) + eps
        dice = (2 * intersection + eps) / denom
        dice_loss = 1.0 - dice
        ################################################################################################################

        return dice_loss

    def focal_loss(
        self,
        category_predictions: torch.Tensor,     # (bsz * fpn * S^2, C - 1,)
        category_ground_truths: torch.Tensor,   # (bsz * fpn * S^2,)
    ) -> torch.Tensor:                          # (bsz * fpn * S^2,)
        """
        Compute the focal loss.
        """
        ################################################################################################################
        # TODO: Compute the focal loss
        ################################################################################################################
        # Build one-hot targets for positives; background is encoded as -1 and should produce zeros
        num_classes = category_predictions.shape[1]
        targets = category_ground_truths.clone()
        # Map background (<0) to an ignore mask
        pos_mask = targets >= 0
        clamped = torch.clamp(targets, min=0)
        y = F.one_hot(clamped, num_classes=num_classes).to(category_predictions.dtype)
        y = y * pos_mask.unsqueeze(1).to(category_predictions.dtype)

        p = torch.clamp(category_predictions, 1e-6, 1 - 1e-6)
        alpha = self.cate_loss_cfg.get("alpha", 0.25)
        gamma = self.cate_loss_cfg.get("gamma", 2.0)

        # Focal loss for multi-label style per-class targets
        pt = torch.where(y == 1, p, 1 - p)
        alpha_t = torch.where(y == 1, torch.full_like(p, alpha), torch.full_like(p, 1 - alpha))
        loss = -alpha_t * (1 - pt).pow(gamma) * torch.log(pt)
        focal_loss = loss.sum(dim=1)  # sum over classes
        ################################################################################################################

        return focal_loss

    def process_predictions(
        self,
        instance_prediction_list: list[torch.Tensor],   # fpn x (bsz, S^2, ori_H / 4, ori_W / 4)
        category_prediction_list: list[torch.Tensor],   # fpn x (bsz, C - 1, S, S)
        original_size: tuple[int, int]                  # (ori_H, ori_W)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process the predictions.
        """
        upsample_size = (original_size[0] // 4, original_size[1] // 4)
        instance_prediction_list = [F.interpolate(t, size=upsample_size, mode="bilinear", align_corners=False) for t in instance_prediction_list]
        return (*zip(*[
            self.process_predictions_for_single_image(ins_pred_img, cate_pred_img, original_size)
            for ins_pred_img, cate_pred_img in zip(
                zip(*instance_prediction_list),
                zip(*category_prediction_list),
            )
        ]),)

    def process_predictions_for_single_image(
        self,
        instance_predictions: list[torch.Tensor],   # fpn x (S^2, ori_H / 4, ori_W / 4)
        category_predictions: list[torch.Tensor],   # fpn x (C - 1, S, S)
        original_size: tuple[int, int]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process the predictions for a single image.
        Input:
        - instance_predictions (list[torch.FloatTensor]): Shape fpn x (S^2, original_H / 4, original_W / 4)
        - category_predictions (list[torch.LongTensor]): Shape fpn x (C - 1, S, S)
        - original_size (tuple[int, int]): (original_H, original_W)
        Output:
        - scores (torch.FloatTensor): Shape (n_obj)
        - indices (torch.LongTensor): Shape (n_obj)
        - masks (torch.FloatTensor): Shape (n_obj, original_H, original_W)
        """

        ################################################################################################################
        # TODO: Process the model predictions at inference time by applying points-NMS, confidence threshold, and
        #  matrix-NMS. Output at most `keep_instance` predictions
        ################################################################################################################
        # Apply points-NMS on categories
        filtered_categories = []
        for cate in category_predictions:
            cate_nms = self.points_nms(cate.unsqueeze(0)).squeeze(0)
            filtered_categories.append(cate_nms)

        candidate_scores = []
        candidate_labels = []
        candidate_masks = []

        for level_idx, (ins_pred, cate_pred) in enumerate(zip(instance_predictions, filtered_categories)):
            S = cate_pred.shape[-1]
            # Flatten grid
            cate_scores, cate_labels = torch.max(cate_pred, dim=0)
            # Keep positions above threshold
            keep = cate_scores >= self.postprocess_cfg["cate_thresh"]
            if keep.sum() == 0:
                continue
            ys, xs = torch.where(keep)
            for y_idx, x_idx in zip(ys, xs):
                grid_idx = (y_idx * S + x_idx).item()
                cls = cate_labels[y_idx, x_idx].item()
                score = cate_scores[y_idx, x_idx].item()
                mask = ins_pred[grid_idx]
                candidate_scores.append(score)
                candidate_labels.append(cls)
                candidate_masks.append(mask)

        if len(candidate_scores) == 0:
            return (
                torch.zeros(0, device=instance_predictions[0].device),
                torch.zeros(0, dtype=torch.long, device=instance_predictions[0].device),
                torch.zeros(0, *instance_predictions[0].shape[-2:], device=instance_predictions[0].device),
            )

        scores = torch.tensor(candidate_scores, device=instance_predictions[0].device)
        labels = torch.tensor(candidate_labels, dtype=torch.long, device=instance_predictions[0].device)
        masks = torch.stack(candidate_masks, dim=0)

        # Matrix NMS score decay
        order = torch.argsort(scores, descending=True)
        masks_sorted = masks[order]
        scores_sorted = scores[order]
        decayed = self.matrix_nms(masks_sorted, scores_sorted, method="gauss", gauss_sigma=0.5)
        # Threshold and keep top-k
        keep_mask = decayed >= self.postprocess_cfg["ins_thresh"]
        keep_idx = torch.nonzero(keep_mask, as_tuple=True)[0]
        if keep_idx.numel() > self.postprocess_cfg["keep_instance"]:
            keep_idx = keep_idx[: self.postprocess_cfg["keep_instance"]]
        masks_kept = masks_sorted[keep_idx]
        scores_kept = decayed[keep_idx]
        labels_kept = labels[order][keep_idx]

        # Binarize masks
        binary_masks = (masks_kept >= self.postprocess_cfg["ins_thresh"]).float()

        return scores_kept, labels_kept, binary_masks
        ################################################################################################################

    def matrix_nms(
        self,
        sorted_masks: torch.Tensor,     # (n_predictions, H, W)
        sorted_scores: torch.Tensor,    # (n_predictions)
        method: str = "gauss",
        gauss_sigma: float = 0.5,
    ) -> torch.Tensor:                  # (n_predictions)
        ################################################################################################################
        # TODO: Implement matrix NMS
        ################################################################################################################
        if sorted_masks.numel() == 0:
            return sorted_scores
        # Binarize masks for IoU computation
        m = (sorted_masks >= 0.5).float().reshape(sorted_masks.shape[0], -1)
        inter = m @ m.t()
        areas = m.sum(dim=1, keepdim=True)
        union = areas + areas.t() - inter + 1e-6
        iou = inter / union
        # Only consider higher-scored masks
        upper = torch.triu(iou, diagonal=1)
        iou_max, _ = upper.max(dim=0)
        if method == "gauss":
            decay = torch.exp(- (iou_max ** 2) / (gauss_sigma + 1e-6))
        else:
            decay = 1 - iou_max
        decayed_scores = sorted_scores * decay
        ################################################################################################################

        return decayed_scores




