"""Pretrained EuroSAT models for the five Station 1 experiments."""

import math
from pathlib import Path

import timm
import torch
from safetensors.torch import load_file
from torch import nn
from torch.nn.utils import parametrize
from torchvision.models import ResNet18_Weights
from torchvision.models.resnet import BasicBlock, ResNet


CACHE = Path(__file__).resolve().parents[1] / ".cache" / "station1"
VIT = "vit_small_patch16_224.augreg_in21k_ft_in1k"
LORA_RANKS = {"vit_lora_r4": 4, "vit_lora_r8": 8, "vit_lora_r16": 16}


def input_config(architecture, *, center_crop=False):
    """Read the checkpoint's input size, interpolation and channel statistics."""
    if architecture == "resnet18":
        preset = ResNet18_Weights.IMAGENET1K_V1.transforms()
        size = [preset.crop_size[0]] * 2
        mode, mean, std = preset.interpolation.value, preset.mean, preset.std
        short_edge = preset.resize_size[0]
    else:
        cfg = timm.get_pretrained_cfg(architecture)
        size = list(cfg.input_size[1:])
        mode, mean, std = cfg.interpolation, cfg.mean, cfg.std
        short_edge = int(min(size) / cfg.crop_pct)
    
    config = dict(size=size, interpolation=mode, mean=list(mean), std=list(std))
    
    if center_crop:
        config["short_edge"] = short_edge
    
    return config


class FrozenResNet18(ResNet):
    """ResNet18 with fixed ImageNet features, including BatchNorm statistics."""

    def __init__(self):
        super().__init__(BasicBlock, [2, 2, 2, 2])

    def train(self, mode=True):
        super().train(mode)
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        return self


class LoRAWeight(nn.Module):
    """Add B @ A to the frozen QKV weight with alpha equal to rank."""

    def __init__(self, projection, rank):
        super().__init__()
        self.A = nn.Parameter(projection.weight.new_empty(rank, projection.in_features))
        self.B = nn.Parameter(projection.weight.new_zeros(projection.out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, weight):
        return weight + self.B @ self.A


def build_model(experiment, num_classes=10, *, pretrained=True, vit_arch=VIT):
    """Build a Station 1 model. Disable pretrained loading when restoring best.pt."""
    
    if experiment == "cnn":
        model = FrozenResNet18()
        checkpoint = CACHE / "resnet18-f37072fd.pth"
    
        if pretrained:
    
            if checkpoint.is_file():
                weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
            else:
                weights = ResNet18_Weights.IMAGENET1K_V1.get_state_dict(progress=True, check_hash=True)
            model.load_state_dict(weights)
    
        model.requires_grad_(False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    
        return model.train()

    if experiment != "vit_linear" and experiment not in LORA_RANKS:
        raise ValueError(f"Unknown Station 1 experiment: {experiment}")

    cached_name = {
        VIT: "vit_small.safetensors",
        "vit_tiny_patch16_224.augreg_in21k_ft_in1k": "vit_tiny.safetensors"}.get(vit_arch)
    
    checkpoint = CACHE / cached_name if cached_name else None
    
    use_cache = pretrained and checkpoint is not None and checkpoint.is_file()
    model = timm.create_model(vit_arch, pretrained=pretrained and not use_cache)
    
    if use_cache:
        model.load_state_dict(load_file(str(checkpoint)))
    
    model.requires_grad_(False)
    model.reset_classifier(num_classes)
    
    if experiment in LORA_RANKS:
        for block in model.blocks:
            qkv = block.attn.qkv
            adapter = LoRAWeight(qkv, LORA_RANKS[experiment])
            parametrize.register_parametrization(qkv, "weight", adapter)
    
    return model
