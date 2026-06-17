"""
SSFTM: Spectral State Fusion Tree Mamba for Hyperspectral Image Classification.

Minimal entry point for reproduction: load a hyperspectral dataset, train the
model, and report Overall Accuracy (OA), Average Accuracy (AA) and Kappa.
Set GENERATE_CLS_MAP=True to additionally save the predicted classification maps.
"""

import os
import random
from operator import truediv

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

import get_cls_map
from model import GrootV


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET_NAME = "HU18"        # IP / PU / SA / HU13 / HU18 / HH / HC / LK
PATCH_SIZE = 5
PCA_COMPONENTS = 20
TEST_RATIO = 0.99            # fraction of labeled samples used for testing
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 1e-3
SEED = 0
GENERATE_CLS_MAP = False     # if True, save classification maps under ./pic

# Datasets are read from this directory. Override without editing code:
#   export SSFTM_DATA_ROOT=/path/to/data
DATA_ROOT = os.environ.get("SSFTM_DATA_ROOT", "./data")

DATASETS = {
    "IP":   {"data_path": "Indian_Pines/Indian_pines_corrected.mat", "data_key": "indian_pines_corrected",
             "gt_path": "Indian_Pines/Indian_pines_gt.mat",          "gt_key": "indian_pines_gt"},
    "PU":   {"data_path": "Pavia University/PaviaU.mat",             "data_key": "paviaU",
             "gt_path": "Pavia University/PaviaU_gt.mat",            "gt_key": "paviaU_gt"},
    "SA":   {"data_path": "Salinas/Salinas_corrected.mat",          "data_key": "salinas_corrected",
             "gt_path": "Salinas/Salinas_gt.mat",                   "gt_key": "salinas_gt"},
    "HU13": {"data_path": "Houston 2013/HustonU_IM.mat",            "data_key": "hustonu",
             "gt_path": "Houston 2013/HustonU_gt.mat",              "gt_key": "hustonu_gt"},
    "HU18": {"data_path": "Houston 2018/houstonU2018.mat",          "data_key": "houstonU",
             "gt_path": "Houston 2018/houstonU2018.mat",            "gt_key": "houstonU_gt"},
    "HH":   {"data_path": "HongHu/WHU_Hi_HongHu.mat",               "data_key": "WHU_Hi_HongHu",
             "gt_path": "HongHu/WHU_Hi_HongHu_gt.mat",              "gt_key": "WHU_Hi_HongHu_gt"},
    "HC":   {"data_path": "HanChuan/WHU_Hi_HanChuan.mat",           "data_key": "WHU_Hi_HanChuan",
             "gt_path": "HanChuan/WHU_Hi_HanChuan_gt.mat",          "gt_key": "WHU_Hi_HanChuan_gt"},
    "LK":   {"data_path": "LongKou/WHU_Hi_LongKou.mat",             "data_key": "WHU_Hi_LongKou",
             "gt_path": "LongKou/WHU_Hi_LongKou_gt.mat",            "gt_key": "WHU_Hi_LongKou_gt"},
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------------
# Data loading and preprocessing
# ---------------------------------------------------------------------------
def _load_mat(path, key):
    mat = sio.loadmat(path)
    if key not in mat:
        available = [k for k in mat if not k.startswith("__")]
        raise KeyError(f"Key '{key}' not found in {path}. Available: {available}")
    return mat[key]


def load_data(name):
    cfg = DATASETS[name]
    data = _load_mat(os.path.join(DATA_ROOT, cfg["data_path"]), cfg["data_key"])
    gt = _load_mat(os.path.join(DATA_ROOT, cfg["gt_path"]), cfg["gt_key"])
    return data, gt


def apply_pca(x, n_components):
    h, w, c = x.shape
    x = x.reshape(-1, c)
    x = PCA(n_components=n_components, whiten=True, random_state=0).fit_transform(x)
    return x.reshape(h, w, n_components)


def extract_patches(x, window):
    """Extract a (window x window) patch centered on every pixel.

    Returns an array of shape (H*W, window, window, C) in row-major order.
    """
    margin = (window - 1) // 2
    x = np.pad(x, ((margin, margin), (margin, margin), (0, 0)), mode="constant")
    patches = sliding_window_view(x, (window, window, x.shape[2]))[:, :, 0]
    return patches.reshape(-1, window, window, x.shape[2])


def to_tensor(patches):
    """(N, H, W, C) -> float tensor (N, C, H, W)."""
    patches = np.ascontiguousarray(patches.transpose(0, 3, 1, 2))
    return torch.from_numpy(patches).float()


def build_loaders(seed):
    data, gt = load_data(DATASET_NAME)
    print(f"Dataset: {DATASET_NAME} | data {data.shape} | gt {gt.shape}")

    data = apply_pca(data, PCA_COMPONENTS)
    patches_all = extract_patches(data, PATCH_SIZE)            # (H*W, P, P, C)
    labels_all = gt.reshape(-1).astype(np.int64)              # 0 = background

    mask = labels_all > 0
    patches = patches_all[mask]
    labels = labels_all[mask] - 1                             # shift to 0-based
    num_classes = int(labels.max()) + 1

    x_train, _, y_train, _ = train_test_split(
        patches, labels, test_size=TEST_RATIO, random_state=seed, stratify=labels
    )

    train_loader = DataLoader(
        TensorDataset(to_tensor(x_train), torch.from_numpy(y_train)),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=True, pin_memory=True,
    )
    # evaluate on all labeled samples
    test_loader = DataLoader(
        TensorDataset(to_tensor(patches), torch.from_numpy(labels)),
        batch_size=BATCH_SIZE, shuffle=False, pin_memory=True,
    )

    loaders = {"train": train_loader, "test": test_loader, "num_classes": num_classes}

    if GENERATE_CLS_MAP:
        # extra loaders over labeled-only and whole-image patches, in pixel order
        loaders["labeled"] = test_loader
        loaders["whole"] = DataLoader(
            TensorDataset(to_tensor(patches_all), torch.from_numpy(labels_all)),
            batch_size=BATCH_SIZE, shuffle=False,
        )
        loaders["gt"] = gt
    return loaders


# ---------------------------------------------------------------------------
# Train and test
# ---------------------------------------------------------------------------
def train(train_loader, num_classes, device):
    net = GrootV(
        act_layer="SiLU",
        norm_layer="LN",
        in_chans=PCA_COMPONENTS,
        num_classes=num_classes,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        net.train()
        running_loss, n_steps = 0.0, 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(net(data), target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_steps += 1
        print(f"[Epoch {epoch + 1:3d}/{EPOCHS}] loss = {running_loss / max(n_steps, 1):.4f}")

    return net


@torch.no_grad()
def test(net, test_loader, device):
    net.eval()
    y_pred, y_true = [], []
    for data, target in test_loader:
        out = net(data.to(device))
        y_pred.append(out.argmax(dim=1).cpu().numpy())
        y_true.append(target.numpy())
    return np.concatenate(y_pred), np.concatenate(y_true)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def average_accuracy(conf):
    each_acc = np.nan_to_num(truediv(np.diag(conf), np.sum(conf, axis=1)))
    return each_acc, np.mean(each_acc)


def report(y_true, y_pred):
    oa = accuracy_score(y_true, y_pred)
    conf = confusion_matrix(y_true, y_pred)
    each_acc, aa = average_accuracy(conf)
    kappa = cohen_kappa_score(y_true, y_pred)
    return oa * 100, aa * 100, kappa * 100, each_acc * 100


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    set_seed(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    loaders = build_loaders(SEED)
    net = train(loaders["train"], loaders["num_classes"], device)
    y_pred, y_true = test(net, loaders["test"], device)

    oa, aa, kappa, each_acc = report(y_true, y_pred)
    print(f"\nOA = {oa:.2f} | AA = {aa:.2f} | Kappa = {kappa:.2f}")
    print("Per-class accuracy (%): " + np.array2string(each_acc, precision=2))

    if GENERATE_CLS_MAP:
        get_cls_map.get_cls_map(net, device, loaders["labeled"], loaders["whole"], loaders["gt"])


if __name__ == "__main__":
    main()
