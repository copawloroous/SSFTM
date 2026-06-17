<div align="center">

# SSFTM

### Spectral State Fusion Tree Mamba for Hyperspectral Image Classification

[![Paper](https://img.shields.io/badge/Paper-IEEE%20TIP%202026-blue)](https://doi.org/10.1109/TIP.2026.3700929)
[![DOI](https://img.shields.io/badge/DOI-10.1109%2FTIP.2026.3700929-orange)](https://doi.org/10.1109/TIP.2026.3700929)
[![Code](https://img.shields.io/badge/Code-GitHub-black)](https://github.com/copawloroous/SSFTM)

---

[Bing Tu](https://scholar.google.com/citations?user=iMuSewsAAAAJ)<sup>1,2,3,4,5 *</sup> &nbsp;|&nbsp;
[Zhenghao Hu](https://ieeexplore.ieee.org/author/721998129448425)<sup>1,2,3,4,5</sup> &nbsp;|&nbsp;
[Bo Liu](https://ieeexplore.ieee.org/author/37404906400)<sup>1,2,3,4,5</sup> &nbsp;|&nbsp;
[Yan He](https://ieeexplore.ieee.org/author/279730212927568)<sup>1,2,3,4,5</sup>

> **Published in:** *IEEE Transactions on Image Processing* (IEEE TIP, Early Access, 2026)

</div>

<details>
<summary><b>📍 Author Affiliations</b> (click to expand)</summary>

<sup>1</sup> Institute of Optics and Electronics  
<sup>2</sup> State Key Laboratory Cultivation Base of Atmospheric Optoelectronic Detection and Information Fusion  
<sup>3</sup> Jiangsu International Joint Laboratory on Meteorological Photonics and Optoelectronic Detection  
<sup>4</sup> Jiangsu Engineering Research Center for Intelligent Optoelectronic Sensing Technology of Atmosphere  
<sup>5</sup> Nanjing University of Information Science and Technology, Nanjing 210044, China  
<sup>*</sup> Corresponding author

</details>

---

> 🤗 **Should you encounter any issues, feel free to contact the author at any time!**
> If this project helps you, please give it a ⭐ — your support means a lot!

---

## 📰 News

- **2026-06**: 🎉 Our work **"Spectral State Fusion Tree Mamba for Hyperspectral Image Classification"** has been accepted by **IEEE TIP** (*IEEE Transactions on Image Processing*, 中科院一区 TOP / JCR Q1, CCF-A, IF 13.7)!
- **2025-03**: 🎉 Our previous work **"Self-Supervised Graph Masked Autoencoders for Hyperspectral Image Classification"** ([SGMAE](https://github.com/copawloroous/SGMAE)) has been accepted by **IEEE TGRS**!

---

## 🔍 Overview

Mamba-style state-space models bring linear-complexity long-range modeling to hyperspectral
image (HSI) classification, but the conventional fixed scan order imposes an arbitrary
spatial ordering and processes each spectral channel independently. **SSFTM** addresses both
issues with two components:

- **Tree Scan (TS).** Cosine distances among neighboring pixels and among spectral channels
  are used to build adaptive minimum spanning trees in **both** the spatial and spectral
  domains, so the scan path follows feature similarity rather than a fixed raster order.
- **Spectral State Fusion (SSF).** Multi-layer 1-D dilated convolutions are applied along the
  spectral dimension of the state-space vectors, enabling inter-channel interaction and
  multi-scale spectral feature extraction.

Together they establish reasonable spatial–spectral relationships and enable efficient joint
feature extraction with acceptable computational cost.

---

## 🛠️ Environment Requirements

| Library      | Version |
|--------------|---------|
| Python       | 3.9     |
| PyTorch      | 1.13.1 (cu117) |
| einops       | 0.7.0   |
| timm         | 0.6.11  |
| scikit-learn | 1.5.2   |
| scipy        | 1.13.0  |
| matplotlib   | 3.8.3   |

```bash
conda create -n SSFTM python=3.9
conda activate SSFTM

# PyTorch (CUDA 11.7 build used in the paper)
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 \
    --extra-index-url https://download.pytorch.org/whl/cu117

pip install -r requirements.txt
```

> **Note on the `_C` extension.** The tree-scanning core is shipped as a precompiled
> extension (`_C.cpython-39-x86_64-linux-gnu.so`) built for **Python 3.9 / x86_64 Linux**.
> If your environment differs, rebuild it from the C/CUDA source before running.

---

## 🚀 Usage

All settings live in the **Configuration** block at the top of [`main.py`](main.py):

| Option | Meaning |
|--------|---------|
| `DATASET_NAME` | `IP / PU / SA / HU13 / HU18 / HH / HC / LK` |
| `PATCH_SIZE`, `PCA_COMPONENTS` | input patch size and number of PCA bands |
| `TEST_RATIO` | fraction of labeled samples used for testing |
| `EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`, `SEED` | training schedule |
| `GENERATE_CLS_MAP` | set `True` to save classification maps under `./pic` |

Then run:

```bash
python main.py
```

The script prints OA / AA / Kappa and the per-class accuracy. If `GENERATE_CLS_MAP=True`,
the predicted maps (`pred_1`, `pred_2`, `gt`) are saved to `./pic` via
[`get_cls_map.py`](get_cls_map.py).

### Data preparation

Place the `.mat` files under `data/` (or set `export SSFTM_DATA_ROOT=/path/to/data`):

```
data/
├── Indian_Pines/    Indian_pines_corrected.mat, Indian_pines_gt.mat
├── Pavia University/ PaviaU.mat, PaviaU_gt.mat
├── Salinas/          Salinas_corrected.mat, Salinas_gt.mat
├── Houston 2013/     HustonU_IM.mat, HustonU_gt.mat
├── Houston 2018/     houstonU2018.mat
├── HongHu/           WHU_Hi_HongHu.mat, WHU_Hi_HongHu_gt.mat
├── HanChuan/         WHU_Hi_HanChuan.mat, WHU_Hi_HanChuan_gt.mat
└── LongKou/          WHU_Hi_LongKou.mat, WHU_Hi_LongKou_gt.mat
```

The common public sources are GIC (`ehu.eus`) for Indian Pines / Pavia / Salinas, the IEEE
GRSS Data Fusion Contest for Houston, and RSIDEA (Wuhan University) for the WHU-Hi scenes.

---

## 👥 Authors

### Zhenghao Hu

🎓 **Education**

- **B.Eng.** in Optoelectronic Information Science and Engineering,
  Nanjing University of Information Science and Technology, China
- **Incoming Ph.D. Student** in Pattern Recognition and Intelligent Systems,
  Institute of Automation, Chinese Academy of Sciences, China

🔬 **Research Interests:** Machine Learning · Computer Vision · Pattern Recognition · Hyperspectral Image Processing

📫 **Contact:** [huzhenghao2026@ia.ac.cn](mailto:huzhenghao2026@ia.ac.cn)

🔗 **Profiles**  
[![Google Scholar](https://img.shields.io/badge/Google%20Scholar-4285F4?logo=google-scholar&logoColor=white)](https://scholar.google.com/citations?user=F5Qx7kAAAAAJ&hl=zh-CN&oi=sra)
[![GitHub](https://img.shields.io/badge/GitHub-181717?logo=github&logoColor=white)](https://github.com/copawloroous)
[![ORCID](https://img.shields.io/badge/ORCID-A6CE39?logo=orcid&logoColor=white)](https://orcid.org/0009-0004-0285-5763)
[![IEEE](https://img.shields.io/badge/IEEE-00629B?logo=ieee&logoColor=white)](https://ieeexplore.ieee.org/author/721998129448425)

---

### Prof. Bing Tu — *Corresponding Author*

🎓 Professor and Ph.D. Supervisor,
*School of Physics and Optoelectronic Engineering*,
Nanjing University of Information Science and Technology, China

🔗 **Profiles**  
[![Google Scholar](https://img.shields.io/badge/Google%20Scholar-4285F4?logo=google-scholar&logoColor=white)](https://scholar.google.com/citations?user=iMuSewsAAAAJ&hl=zh-CN&oi=sra)
[![Faculty Page](https://img.shields.io/badge/Faculty%20Profile-NUIST-1e3a8a)](https://faculty.nuist.edu.cn/tubing/zh_CN/index.htm)

---

## 📖 Citation

If you find this code useful in your research, please cite our paper:

```bibtex
@article{tu2026spectral,
  title={Spectral State Fusion Tree Mamba for Hyperspectral Image Classification},
  author={Tu, Bing and Hu, Zhenghao and Liu, Bo and He, Yan},
  journal={IEEE Transactions on Image Processing},
  year={2026},
  publisher={IEEE}
}
```
