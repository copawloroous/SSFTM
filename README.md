<div align="center">

# SSFTM

### Spectral State Fusion Tree Mamba for Hyperspectral Image Classification

[![Paper](https://img.shields.io/badge/Paper-IEEE%20TIP%202026-blue)](https://doi.org/10.1109/TIP.2026.3700929)
[![Code](https://img.shields.io/badge/Code-GitHub-black)](https://github.com/copawloroous/SSFTM)

---

[Bing Tu](https://scholar.google.com/citations?user=iMuSewsAAAAJ)<sup>1,2,3,4,5 *</sup> &nbsp;|&nbsp;
[Zhenghao Hu](https://ieeexplore.ieee.org/author/721998129448425)<sup>1,2,3,4,5</sup> &nbsp;|&nbsp;
[Bo Liu](https://ieeexplore.ieee.org/author/37404906400)<sup>1,2,3,4,5</sup> &nbsp;|&nbsp;
[Yan He](https://ieeexplore.ieee.org/author/279730212927568)<sup>1,2,3,4,5</sup>

> **Published in:** *IEEE Transactions on Image Processing* (IEEE TIP 2026)

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

- **2026-06**: 🎉 Our work **"Spectral State Fusion Tree Mamba for Hyperspectral Image Classification"** has been published in **IEEE TIP** (*IEEE Transactions on Image Processing*, 中科院一区 TOP / JCR Q1, CCF-A, IF 15.3)!
- **2025-03**: 🎉 Our previous work **"Self-Supervised Graph Masked Autoencoders for Hyperspectral Image Classification"** ([SGMAE](https://github.com/copawloroous/SGMAE)) has been accepted by **IEEE TGRS**(*IEEE Transactions on Geoscience and Remote Sensing*, 中科院一区 TOP / JCR Q1, CCF-B, IF 8.6)!

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

> **Note:** The tree-scanning core (Boruvka MST, BFS, tree-scan refine) ships as a precompiled extension (`_C.cpython-39-x86_64-linux-gnu.so`) built for **Python 3.9 / x86_64 Linux**. Rebuild it from source if your environment differs.

---

## 🚀 Usage Instructions

1. **Dataset Preparation**  
   The paper uses four benchmarks:

   | Benchmark | Size | Bands | Classes | Training samples |
   |:---------:|:----------:|:-----:|:-------:|:----------------:|
   | **HU13** (Houston 2013) | 349 × 1905 | 144 | 15 | 1 % per class (156) |
   | **HU18** (Houston 2018) | 601 × 2384 | 50 | 20 | 1 % per class (5,039) |
   | **HC** (Han Chuan) | 1217 × 303 | 274 | 16 | 20 per class (300) |
   | **HH** (Hong Hu) | 940 × 475 | 270 | 22 | 20 per class (440) |

   Download the Houston scenes from the IEEE GRSS Data Fusion Contest (2013 / 2018) and the WHU-Hi scenes (HC / HH) from RSIDEA, Wuhan University. Place the `.mat` files under `./data` (or `export SSFTM_DATA_ROOT=/path/to/data`):

   ```text
   data/
   ├── Houston 2013/  HustonU_IM.mat, HustonU_gt.mat
   ├── Houston 2018/  houstonU2018.mat
   ├── HongHu/        WHU_Hi_HongHu.mat, WHU_Hi_HongHu_gt.mat
   └── HanChuan/      WHU_Hi_HanChuan.mat, WHU_Hi_HanChuan_gt.mat
   ```

   `IP / PU / SA / LK` are also supported by the code (see `DATASETS` in [`main.py`](main.py)) but are not used in the paper.

2. **Configuration**  
   All settings live in the **Configuration** block of [`main.py`](main.py). Per-benchmark settings used in the paper:

   | Benchmark | `DATASET_NAME` | `PCA_COMPONENTS` | `PATCH_SIZE` | `TEST_RATIO` |
   |:---------:|:--------------:|:----------------:|:------------:|:------------:|
   | Houston 2013 | `"HU13"` | 20 | 5 | 0.99 |
   | Houston 2018 | `"HU18"` | 20 | 5 | 0.99 |
   | Han Chuan | `"HC"` | 30 | 9 | ≈ 0.9988 |
   | Hong Hu | `"HH"` | 30 | 9 | ≈ 0.9989 |

   `TEST_RATIO` is the fraction of labeled samples reserved for testing; the split is stratified per class, so `1 − TEST_RATIO` follows the paper's per-class sampling protocol (e.g., for HC: `1 − (20 × 16) / 257,530`).

3. **Training & Evaluation**  
   ```bash
   python main.py
   ```
   Trains with Adam + cross-entropy, then prints **OA / AA / Kappa** and the per-class accuracy over all labeled samples.

4. **Classification Maps**  
   Set `GENERATE_CLS_MAP = True` to save the predicted maps (`pred_1`, `pred_2`, `gt`) under `./pic` via [`get_cls_map.py`](get_cls_map.py).

5. **Hardware Recommendation**  
   An **NVIDIA RTX 2080 Ti** (11 GB) was used in the paper. If you run into GPU memory errors, try reducing `PCA_COMPONENTS` or `PATCH_SIZE`.

---

## 🧪 Reference Experiments

> *Results reported in the paper (mean ± std), following the sampling protocols above. SSFTM ranks first in OA / AA / Kappa among all twelve compared CNN / Transformer / Mamba methods.* (Hardware: Intel i9-10900KF · NVIDIA RTX 2080 Ti (11 GB VRAM) · 32 GB RAM.)

| Benchmark | OA (%) | AA (%) | Kappa (%) |
|:---------:|:---------------:|:---------------:|:----------------:|
| **HU13** | **88.58 ± 1.86** | **89.19 ± 1.47** | **87.66 ± 2.01** |
| **HU18** | **89.42 ± 0.10** | **82.69 ± 1.45** | **86.19 ± 0.13** |
| **HC**   | **88.00 ± 0.95** | **87.06 ± 0.73** | **86.06 ± 1.09** |
| **HH**   | **90.18 ± 0.97** | **89.67 ± 0.79** | **87.75 ± 1.18** |

---

## 👥 Authors

### Prof. Bing Tu — *First Author, Advisor & Corresponding Author*

🎓 Professor and Ph.D. Supervisor  
*School of Physics and Optoelectronic Engineering*  
Nanjing University of Information Science and Technology, China

🔗 **Profiles**  
[![Google Scholar](https://img.shields.io/badge/Google%20Scholar-4285F4?logo=google-scholar&logoColor=white)](https://scholar.google.com/citations?user=iMuSewsAAAAJ&hl=zh-CN&oi=sra)
[![Faculty Page](https://img.shields.io/badge/Faculty%20Profile-NUIST-1e3a8a)](https://faculty.nuist.edu.cn/tubing/zh_CN/index.htm)

---

### Zhenghao Hu — *Second Author*

🎓 **Education**

- **B.Eng.** in Optoelectronic Information Science and Engineering  
  *School of Physics and Optoelectronic Engineering*  
  Nanjing University of Information Science and Technology, China

- **Ph.D. Student** in Pattern Recognition and Intelligent Systems  
  *Institute of Automation*  
  Chinese Academy of Sciences, China

🔬 **Research Interests**  
Machine Learning · Computer Vision · Pattern Recognition · Hyperspectral Image Processing

🏛️ **Affiliation**

- The Key Laboratory of Cognition and Decision Intelligence for Complex Systems, Institute of Automation, Chinese Academy of Sciences  
- School of Artificial Intelligence, University of Chinese Academy of Sciences  
- School of Physics and Optoelectronic Engineering, Nanjing University of Information Science and Technology

📖 **Biography**  
Zhenghao Hu (Student Member, IEEE) received the B.Sc.Eng. degree in optoelectronic information science and engineering from Nanjing University of Information Science and Technology, Nanjing, China, in 2026. He is currently pursuing the Ph.D. degree in pattern recognition and intelligent systems with the Institute of Automation, Chinese Academy of Sciences, Beijing, China. His research interests include computer vision, machine learning, and pattern recognition.

📫 **Contact**  
Current: [huzhenghao2026@ia.ac.cn](mailto:huzhenghao2026@ia.ac.cn)  
Previous: ~~[202213880076@nuist.edu.cn](mailto:202213880076@nuist.edu.cn)~~

🔗 **Profiles**  
[![Google Scholar](https://img.shields.io/badge/Google%20Scholar-4285F4?logo=google-scholar&logoColor=white)](https://scholar.google.com/citations?user=F5Qx7kAAAAAJ&hl=zh-CN&oi=sra)
[![GitHub](https://img.shields.io/badge/GitHub-181717?logo=github&logoColor=white)](https://github.com/copawloroous)
[![ORCID](https://img.shields.io/badge/ORCID-A6CE39?logo=orcid&logoColor=white)](https://orcid.org/0009-0004-0285-5763)
[![IEEE](https://img.shields.io/badge/IEEE-00629B?logo=ieee&logoColor=white)](https://ieeexplore.ieee.org/author/721998129448425)

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
