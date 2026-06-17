"""Render predicted classification maps as color images."""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# RGB color table for labels 0..29 (0 is background / black).
_COLOR_LUT = np.array(
    [
        [0, 0, 0], [147, 67, 46], [0, 0, 255], [255, 100, 0], [0, 255, 123],
        [164, 75, 155], [101, 174, 255], [118, 254, 172], [60, 91, 112], [255, 255, 0],
        [255, 255, 125], [255, 0, 255], [100, 0, 255], [0, 172, 254], [0, 255, 0],
        [171, 175, 80], [101, 193, 60], [255, 102, 102], [255, 158, 204], [204, 204, 255],
        [255, 204, 153], [153, 255, 153], [0, 204, 204], [204, 153, 255], [153, 76, 0],
        [0, 76, 153], [76, 153, 0], [76, 0, 153], [153, 0, 76], [204, 76, 0],
    ],
    dtype=np.float32,
) / 255.0


def labels_to_rgb(labels):
    """Map an integer label array to RGB (..., 3) in [0, 1]."""
    labels = np.clip(labels.astype(np.int64), 0, _COLOR_LUT.shape[0] - 1)
    return _COLOR_LUT[labels]


def build_label_map(y_pred, y_gt, fill_mode):
    """Place a 1-D prediction sequence onto a 2-D map.

    fill_mode=1: fill every pixel in raster order (requires len(y_pred) == H*W).
    fill_mode=2: fill only foreground pixels (y_gt > 0); background stays 0.
    """
    h, w = y_gt.shape
    cls_map = np.zeros((h, w), dtype=np.int64)

    if fill_mode == 1:
        return (y_pred + 1).astype(np.int64).reshape(h, w)

    if fill_mode == 2:
        mask = y_gt > 0
        n = int(mask.sum())
        if len(y_pred) < n:
            raise ValueError(f"y_pred has {len(y_pred)} entries but {n} foreground pixels are required.")
        cls_map[mask] = (y_pred[:n] + 1).astype(np.int64)
        return cls_map

    raise ValueError(f"fill_mode must be 1 or 2, got {fill_mode}")


def save_rgb_map(rgb_map, save_path, dpi=300):
    save_path = str(save_path)
    Path(os.path.dirname(save_path) or ".").mkdir(parents=True, exist_ok=True)

    h, w = rgb_map.shape[:2]
    fig, ax = plt.subplots(figsize=(w * 2.0 / dpi, h * 2.0 / dpi), dpi=dpi)
    ax.axis("off")
    ax.imshow(rgb_map)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


@torch.no_grad()
def predict(net, device, data_loader):
    """Return predicted labels (1-D int array) for the whole loader."""
    net.eval()
    preds = []
    for inputs, _ in data_loader:
        logits = net(inputs.to(device))
        preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


def save_map(net, device, data_loader, y_gt, fill_mode, save_path, dpi=300):
    y_pred = predict(net, device, data_loader)
    cls_map = build_label_map(y_pred, y_gt, fill_mode)
    save_rgb_map(labels_to_rgb(cls_map), save_path, dpi=dpi)


def get_cls_map(net, device, labeled_loader, whole_loader, y_gt):
    """Save three maps under ./pic: foreground prediction, full prediction, ground truth."""
    save_map(net, device, labeled_loader, y_gt, fill_mode=2, save_path="./pic/pred_1.png")
    save_map(net, device, whole_loader, y_gt, fill_mode=1, save_path="./pic/pred_2.png")
    save_rgb_map(labels_to_rgb(y_gt.astype(np.int64)), "./pic/gt.png")
    print("Classification maps saved to ./pic")
