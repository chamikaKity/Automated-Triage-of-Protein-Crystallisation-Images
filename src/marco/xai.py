"""Grad-CAM and occlusion sensitivity maps for explaining ResNet predictions.

Grad-CAM (architecture-specific) targets layer4's last conv block - the same block fine-tuned
in the Final ANN, so its heatmaps show exactly the representations that were trained for this
task. Occlusion (model-agnostic) is an independent cross-check: it directly measures how much
covering each region drops the predicted class's score, with no assumption about model
internals.

These explanations show what the model's gradients/sensitivity point to - they do not prove
the model's true reasoning, and should be read as supporting evidence, not proof (see
notebooks/04_xai.ipynb for the fuller discussion and limitations).
"""

import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    """Grad-CAM for a ResNet's layer4 (its last convolutional block)."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None
        self.layer = model.layer4
        self.layer.register_forward_hook(self._save_activations)
        self.layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, img_tensor: torch.Tensor, target_class: int) -> np.ndarray:
        """img_tensor: (1, 3, H, W), on the same device as the model. Returns an (H, W)
        heatmap normalised to [0, 1].

        Requires grad on the input itself (not just layer4's weights) - on the Initial ANN,
        layer4's parameters are frozen (requires_grad=False), so no gradient would otherwise
        reach it at all, regardless of backbone depth or fine-tuning degree.
        """
        self.model.zero_grad()
        img_tensor = img_tensor.clone().requires_grad_(True)
        output = self.model(img_tensor)
        output[0, target_class].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))  # (1, 1, h, w)
        cam = F.interpolate(cam, size=img_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        return (cam - cam_min) / (cam_max - cam_min + 1e-8)


def occlusion_map(model: torch.nn.Module, img_tensor: torch.Tensor, target_class: int,
                   patch_size: int = 32, stride: int = 16) -> np.ndarray:
    """Model-agnostic sensitivity map: how much covering each patch drops the target class's
    predicted probability, averaged over every patch that covers each pixel. Slower than
    Grad-CAM but makes no assumption about model internals - an independent cross-check.
    """
    model.eval()
    with torch.no_grad():
        base_prob = F.softmax(model(img_tensor), dim=1)[0, target_class].item()

    _, _, h, w = img_tensor.shape
    heatmap = np.zeros((h, w), dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.float32)
    fill_value = img_tensor.mean().item()  # grey fill, roughly neutral post-normalisation

    with torch.no_grad():
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                occluded = img_tensor.clone()
                occluded[:, :, y:y + patch_size, x:x + patch_size] = fill_value
                prob = F.softmax(model(occluded), dim=1)[0, target_class].item()
                heatmap[y:y + patch_size, x:x + patch_size] += base_prob - prob
                counts[y:y + patch_size, x:x + patch_size] += 1

    heatmap = np.clip(heatmap / np.maximum(counts, 1), 0, None)
    return heatmap / heatmap.max() if heatmap.max() > 0 else heatmap
