"""ResNet backbone + simple classifier head, for both the Initial and Final ANN.

Two fine-tuning degrees: "frozen" (Initial ANN - only the head trains) and "layer4" (Final
ANN - the backbone's last residual block also trains), the optional 4th hyperparameter -
approved by the lecturer, see docs/lecturer_approval_fine_tuning.md.
"""

import torch.nn as nn
import torchvision

BACKBONES = {
    "resnet18": (torchvision.models.resnet18, torchvision.models.ResNet18_Weights.IMAGENET1K_V1),
    "resnet34": (torchvision.models.resnet34, torchvision.models.ResNet34_Weights.IMAGENET1K_V1),
}

FINE_TUNE_DEGREES = ["frozen", "layer4"]


def build_model(backbone: str, num_classes: int = 4, dropout: float = 0.4,
                 fine_tune_degree: str = "frozen") -> nn.Module:
    """ImageNet-pretrained ResNet with a new, simple head on top.

    The head is deliberately a single Dropout + Linear layer, not a multi-layer MLP - the
    pretrained features are already strong, and a more complex head would add overfitting
    risk on our modest (~10K image) training set for no clear benefit.

    fine_tune_degree="frozen" trains only the head. "layer4" also unfreezes the backbone's
    last residual block, so it can adapt its ImageNet features to crystallisation images.
    """
    if backbone not in BACKBONES:
        raise ValueError(f"unknown backbone {backbone!r}, choose from {list(BACKBONES)}")
    if fine_tune_degree not in FINE_TUNE_DEGREES:
        raise ValueError(f"unknown fine_tune_degree {fine_tune_degree!r}, choose from {FINE_TUNE_DEGREES}")
    constructor, weights = BACKBONES[backbone]
    net = constructor(weights=weights)

    in_features = net.fc.in_features
    net.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))

    for name, param in net.named_parameters():
        param.requires_grad = name.startswith("fc.") or (
            fine_tune_degree == "layer4" and name.startswith("layer4.")
        )

    return net


def freeze_backbone_bn(model: nn.Module, fine_tune_degree: str = "frozen") -> None:
    """Keep the frozen backbone's BatchNorm layers in eval mode after model.train().

    requires_grad=False stops the conv weights from updating, but BatchNorm's running
    mean/var still shift from forward-pass statistics unless the layer itself is in eval
    mode - calling model.train() each epoch resets every submodule to train mode, so this
    needs to be called again right after it. layer4's BatchNorm layers are skipped when
    fine_tune_degree="layer4", since that block is being trained too and should track
    running stats normally, same as the head's.
    """
    for name, module in model.named_modules():
        if not isinstance(module, nn.BatchNorm2d) or name.startswith("fc"):
            continue
        if fine_tune_degree == "layer4" and name.startswith("layer4"):
            continue
        module.eval()


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (trainable_params, total_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total
