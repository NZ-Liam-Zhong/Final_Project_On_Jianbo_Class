import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import OptimizerLRScheduler

from src.backbone import FPNBackbone, Resnet50Backbone, DinoFPNBackbone
from src.solo_head import SOLOHead


class SOLO(pl.LightningModule):
    def __init__(self, backbone: FPNBackbone = None):
        super().__init__()
        self.backbone = Resnet50Backbone(use_pretrained=False) if backbone is None else backbone
        if isinstance(self.backbone, Resnet50Backbone):
            self.head = SOLOHead(num_classes=4, in_channels=256, seg_feat_channels=256)
            print("Using ResNet50 Backbone")
        elif isinstance(self.backbone, DinoFPNBackbone):
            self.head = SOLOHead(num_classes=4, in_channels=256, seg_feat_channels=256)
            print("Using DINO+FPN Backbone")            
        else:
            raise ValueError("Backbone not supported")

    def forward(self, img: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        backbone_output = self.backbone(img)
        return self.head(backbone_output)

    def training_step(self, batch, batch_idx: int):
        if self.trainer.current_epoch in (27, 33):
            self.lr_schedulers().step()

        img, label_list, mask_list, bbox_list = batch
        cate_pred_list, ins_pred_list = self.forward(img)
        ins_gts_list, ins_ind_gts_list, cate_gts_list = self.head.construct_targets(bbox_list, label_list, mask_list, target_sizes=[t.shape[2:] for t in ins_pred_list])

        losses = self.head.loss(cate_pred_list, ins_pred_list, ins_gts_list, ins_ind_gts_list, cate_gts_list)
        loss_names = ("train_total_loss", "train_focal_loss", "train_dice_loss",)
        for loss, loss_name in zip(losses, loss_names):
            self.log(loss_name, loss.item(), on_step=True, prog_bar=True, batch_size=img.shape[0], sync_dist=True)
        return losses[0]

    def validation_step(self, batch, batch_idx: int):
        img, label_list, mask_list, bbox_list = batch
        cate_pred_list, ins_pred_list = self.forward(img)
        ins_gts_list, ins_ind_gts_list, cate_gts_list = self.head.construct_targets(bbox_list, label_list, mask_list, target_sizes=[t.shape[2:] for t in ins_pred_list])

        losses = self.head.loss(cate_pred_list, ins_pred_list, ins_gts_list, ins_ind_gts_list, cate_gts_list)
        loss_names = ("val_total_loss", "val_focal_loss", "val_dice_loss",)
        for loss, loss_name in zip(losses, loss_names):
            self.log(loss_name, loss.item(), on_step=True, prog_bar=True, batch_size=img.shape[0], sync_dist=True)
        return losses[0]

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.SGD(self.head.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[27, 33], gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}




