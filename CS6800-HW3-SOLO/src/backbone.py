import torch
import torch.nn as nn
import torchvision
from torchvision.models import ResNet50_Weights
from torchvision.models.detection.backbone_utils import BackboneWithFPN

import torch.nn.functional as F
from collections import OrderedDict


class FPNBackbone(nn.Module):
    pass


class Resnet50Backbone(FPNBackbone):
    def __init__(self, use_pretrained: bool = False):
        super().__init__()
        # Avoid network download by default to prevent SSL issues; enable via use_pretrained=True if online
        try:
            weights = ResNet50_Weights.IMAGENET1K_V1 if use_pretrained else None
            self.resnet: BackboneWithFPN = torchvision.models.detection.maskrcnn_resnet50_fpn(
                weights_backbone=weights
            ).backbone
        except Exception:
            # Safe offline fallback
            self.resnet: BackboneWithFPN = torchvision.models.detection.maskrcnn_resnet50_fpn(
                weights_backbone=None
            ).backbone

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            return self.resnet.forward(x)


class DinoFPNBackbone(nn.Module):
    """
    DINO (ViT) backbone + multi-level FPN with semantic hierarchy.
    - Grab 4 transformer block outputs (shallow -> deep) from DINO.
    - Apply lateral 1x1 to each, then do top-down fusion.
    - Finally map fused features to target pyramid strides P1..P5: 1/4, 1/8, 1/16, 1/32, 1/64.
    """

    def __init__(
        self,
        dino_name: str = "dino_vits16",
        out_channels: int = 256,
        n_layers: int = 4,
    ):
        super().__init__()
        self.dino = torch.hub.load("facebookresearch/dino:main", dino_name, pretrained=True)
        self.hidden_dim = getattr(self.dino, "embed_dim", out_channels)
        self.patch_size = getattr(self.dino.patch_embed, "patch_size", 16)
        self.n_layers = n_layers
        self.final_mode = "bilinear"

        for p in self.dino.parameters():
            p.requires_grad = False
        self.dino.eval()

        self.lateral = nn.ModuleList([nn.Conv2d(self.hidden_dim, out_channels, 1) for _ in range(n_layers)])

        self.out_p1 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.out_p2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.out_p3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.out_p4 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.out_p5 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.down2 = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
        self.down4 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1),
            nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1),
        )

        self.out_channels = out_channels

    def _tokens_to_map(self, tokens: torch.Tensor, Ht: int, Wt: int) -> torch.Tensor:
        """
        Convert ViT tokens to a 2D feature map of shape (B, C, Ht, Wt).
        """
        if tokens.shape[1] == 1 + Ht * Wt:
            tokens = tokens[:, 1:, :]
        B, N, C = tokens.shape
        fmap = tokens.reshape(B, Ht, Wt, C).permute(0, 3, 1, 2).contiguous()
        return fmap

    def _get_multi_layer_feats(self, x: torch.Tensor):
        """
        Get 4 transformer layer outputs from DINO.
        """
        need_grad = any(p.requires_grad for p in self.dino.parameters())
        ctx = torch.enable_grad if (self.training and need_grad) else torch.no_grad

        with ctx():
            layers = self.dino.get_intermediate_layers(x, n=self.n_layers)
        return layers

    def forward(self, x: torch.Tensor):
        B, _, H, W = x.shape
        ps = self.patch_size
        Ht, Wt = H // ps, W // ps

        layers = self._get_multi_layer_feats(x)
        maps = [self._tokens_to_map(tok, Ht, Wt) for tok in layers]

        lats = [lat(m) for lat, m in zip(self.lateral, maps)]

        td = [None] * self.n_layers
        td[-1] = lats[-1]
        for i in reversed(range(self.n_layers - 1)):
            td[i] = lats[i] + td[i + 1]

        def _resize(feat, size_hw):
            return F.interpolate(feat, size=size_hw, mode=self.final_mode, align_corners=(self.final_mode != "bilinear"))

        size_p1 = (H // 4,  W // 4)
        size_p2 = (H // 8,  W // 8)
        size_p3 = (H // 16, W // 16)

        p2_base = _resize(td[0], size_p2)
        p3_base = _resize(td[1] if self.n_layers >= 2 else td[0], size_p3)
        p4_base = _resize(td[2] if self.n_layers >= 3 else td[-1], size_p3)
        p5_base = _resize(td[3] if self.n_layers >= 4 else td[-1], size_p3)

        p4_lat = self.down2(p4_base)
        p5_lat = self.down4(p5_base)

        p1_lat = _resize(p2_base, size_p1)

        p1 = self.out_p1(p1_lat)
        p2 = self.out_p2(p2_base)
        p3 = self.out_p3(p3_base)
        p4 = self.out_p4(p4_lat)
        p5 = self.out_p5(p5_lat)

        return {
            "p1": p1,  # stride 4
            "p2": p2,  # stride 8
            "p3": p3,  # stride 16
            "p4": p4,  # stride 32
            "p5": p5,  # stride 64
        }




