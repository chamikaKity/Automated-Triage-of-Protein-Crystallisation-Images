"""Build the Q2 architecture diagram (docs/figures/q2_architecture.png).

One-off figure-generation script, not part of the training/eval pipeline - run manually when
the architecture changes. Parameter counts are read directly from torchvision's ResNet18/34
construction (not hand-copied), so they can't silently drift from the real model.
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import torchvision

FIGURES_DIR = Path(__file__).resolve().parent.parent / "docs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

COLOR_FROZEN = "#0072B2"   # always frozen: ImageNet-pretrained, never trained
COLOR_LAYER4 = "#D55E00"   # frozen in the Initial ANN, trainable in the Final ANN
COLOR_HEAD = "#009E73"     # always trainable: the new head


def stage_params(backbone: str) -> dict:
    ctor = {"resnet18": torchvision.models.resnet18, "resnet34": torchvision.models.resnet34}[backbone]
    net = ctor(weights=None)
    return {name: sum(p.numel() for p in mod.parameters()) for name, mod in net.named_children()}


def main() -> None:
    r18 = stage_params("resnet18")
    r34 = stage_params("resnet34")
    head_params = 512 * 4 + 4  # Dropout has none; Linear(512, 4)

    # (label, output shape, resnet18 params, resnet34 params, color)
    stages = [
        ("Input\n224x224x3", "224x224x3", None, None, "white"),
        ("Conv1 7x7/2\nBN+ReLU+MaxPool", "56x56x64", r18["conv1"] + r18["bn1"], r34["conv1"] + r34["bn1"], COLOR_FROZEN),
        ("Layer1\n(2 blocks / 3 blocks)", "56x56x64", r18["layer1"], r34["layer1"], COLOR_FROZEN),
        ("Layer2\n(2 blocks / 4 blocks)", "28x28x128", r18["layer2"], r34["layer2"], COLOR_FROZEN),
        ("Layer3\n(2 blocks / 6 blocks)", "14x14x256", r18["layer3"], r34["layer3"], COLOR_FROZEN),
        ("Layer4\n(2 blocks / 3 blocks)", "7x7x512", r18["layer4"], r34["layer4"], COLOR_LAYER4),
        ("Global Avg Pool", "512", 0, 0, "white"),
        ("Dropout(0.4)\n+ Linear 512->4", "4", head_params, head_params, COLOR_HEAD),
        ("Softmax", "4 class probs", None, None, "white"),
    ]

    fig, ax = plt.subplots(figsize=(16, 4.5))
    box_w, box_h, gap = 1.9, 1.4, 0.35
    x = 0

    for label, shape, p18, p34, color in stages:
        face = color if color != "white" else "#f5f5f5"
        edge = "black" if color == "white" else color
        rect = mpatches.FancyBboxPatch(
            (x, 0), box_w, box_h, boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=1.5, edgecolor=edge, facecolor=face, alpha=0.85 if color != "white" else 1.0,
        )
        ax.add_patch(rect)
        text_color = "white" if color not in ("white",) else "black"
        ax.text(x + box_w / 2, box_h * 0.68, label, ha="center", va="center", fontsize=7.8,
                color=text_color, weight="bold")
        ax.text(x + box_w / 2, box_h * 0.32, shape, ha="center", va="center", fontsize=8, color=text_color)
        if p18 is not None:
            ax.text(x + box_w / 2, -0.28, f"{p18:,}p (18)\n{p34:,}p (34)", ha="center", va="top", fontsize=6.5)
        if x > 0:
            ax.annotate("", xy=(x, box_h / 2), xytext=(x - gap, box_h / 2),
                        arrowprops=dict(arrowstyle="->", lw=1.3))
        x += box_w + gap

    ax.set_xlim(-0.3, x)
    ax.set_ylim(-0.9, box_h + 0.3)
    ax.axis("off")
    ax.set_title("Backbone (ResNet18 / ResNet34, ImageNet-pretrained) + head - "
                  "parameter counts shown as resnet18 / resnet34", fontsize=10)

    legend_handles = [
        mpatches.Patch(color=COLOR_FROZEN, label="Always frozen (ImageNet-pretrained)"),
        mpatches.Patch(color=COLOR_LAYER4, label="layer4: frozen in Initial ANN, trainable in Final ANN"),
        mpatches.Patch(color=COLOR_HEAD, label="Head: always trainable (both Initial and Final ANN)"),
    ]
    ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.12),
              ncol=3, fontsize=8, frameon=False)

    fig.savefig(FIGURES_DIR / "q2_architecture.png", dpi=160, bbox_inches="tight")
    print(f"saved to {FIGURES_DIR / 'q2_architecture.png'}")

    total18 = sum(p18 for _, _, p18, _, _ in stages if p18 is not None)
    total34 = sum(p34 for _, _, _, p34, _ in stages if p34 is not None)
    print(f"resnet18 total (incl. our 4-class head): {total18:,}")
    print(f"resnet34 total (incl. our 4-class head): {total34:,}")


if __name__ == "__main__":
    main()
