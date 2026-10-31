"""ResNet backbone + simple classifier head for the Initial ANN.

Fine-tuning depth (unfreezing layers beyond the head) is pending lecturer approval - see
docs/. Until that's recorded, this module only builds the frozen-backbone configuration.
"""

import torch.nn as nn
import torchvision

BACKBONES = {
    "resnet18": (torchvision.models.resnet18, torchvision.models.ResNet18_Weights.IMAGENET1K_V1),
    "resnet34": (torchvision.models.resnet34, torchvision.models.ResNet34_Weights.IMAGENET1K_V1),
}


def build_model(backbone: str, num_classes: int = 4, dropout: float = 0.4) -> nn.Module:
    """ImageNet-pretrained ResNet with its backbone frozen and a new, simple head.

    The head is deliberately a single Dropout + Linear layer, not a multi-layer MLP - the
    pretrained features are already strong, and a more complex head would add overfitting
    risk on our modest (~10K image) training set for no clear benefit.
    """
    if backbone not in BACKBONES:
        raise ValueError(f"unknown backbone {backbone!r}, choose from {list(BACKBONES)}")
    constructor, weights = BACKBONES[backbone]
    net = constructor(weights=weights)

    in_features = net.fc.in_features
    net.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))

    for name, param in net.named_parameters():
        if not name.startswith("fc."):
            param.requires_grad = False

    return net


def freeze_backbone_bn(model: nn.Module) -> None:
    """Keep the frozen backbone's BatchNorm layers in eval mode after model.train().

    requires_grad=False stops the conv weights from updating, but BatchNorm's running
    mean/var still shift from forward-pass statistics unless the layer itself is in eval
    mode - calling model.train() each epoch resets every submodule to train mode, so this
    needs to be called again right after it.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d) and not name.startswith("fc"):
            module.eval()


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (trainable_params, total_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total
