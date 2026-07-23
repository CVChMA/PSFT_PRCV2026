<div align="center">
<h1>Point-Selection Fine-Tuning Framework for
Robust Point Cloud Classification (PRCV 2026)</h1>

<a href="https://arxiv.org/pdf/2607.19711" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/arXiv-2607.19711-B31B1B?logo=arxiv&logoColor=white&style=flat-square" alt="arXiv"></a>
<a href="./PSFT_main.pdf" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/Paper-PDF-b31b1b" alt="Paper"></a>
<a href="./PSFT_supp.pdf" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/Supplement-PDF-orange" alt="Supplement"></a>
<a href="https://huggingface.co/SZUChangMa/PSFT" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/Checkpoints-Hugging%20Face-blue" alt="Hugging Face"></a>
<a href="https://huggingface.co/SZUChangMa/PSFT" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/License-MIT-green" alt="License"></a>

<p>
  <span class="author">Da Li</span><sup>1,2&dagger;</sup>,
  <span class="author">Chang Ma</span><sup>2&dagger;</sup>,
  and <span class="author">Dongfu Yin</span><sup>1*</sup>
</p>

<p>
  <sup>1</sup>Guangdong Laboratory of Artificial Intelligence and Digital Economy (Shenzhen), Shenzhen, China<br>
  <sup>2</sup>Shenzhen University, Shenzhen, China
</p>

<p>
  <sup>&dagger;</sup>Equal contribution. <sup>*</sup>Corresponding author.<br>
  li944104439@gmail.com; 2510235035@mails.szu.edu.cn; yindongfu@gml.ac.cn
</p>

<p>
  Official implementation and checkpoints for PSFT on point-cloud corruption benchmarks.
</p>
</div>

## Pretrained Models

The released checkpoints are hosted on [Hugging Face](https://huggingface.co/SZUChangMa/PSFT). Each checkpoint contains the classifier model and point-selection model used by `test_ps.py`.

ModelNet-C and ModelNet40-C share the same checkpoints. Use the `ModelNet-C` checkpoint files for both `ModelNet-C` and `ModelNet40-C` evaluation.

| Dataset | Backbone | Method | Augmentation | Download |
| :--- | :--- | :--- | :--- | :--- |
| `ModelNet-C` / `ModelNet40-C` | `Point-BERT` | PSFT | None | [Link](https://huggingface.co/SZUChangMa/PSFT/blob/main/ModelNet-C/Train-PS-Point-BERT-PG-ModelNet-C-add_FFM-Augmentation%253ANone-train_alpha%253A0.5_epoch_300.pt) |
| `ModelNet-C` / `ModelNet40-C` | `Point-BERT` | PSFT + Aug. | WOLFMix | [Link](https://huggingface.co/SZUChangMa/PSFT/blob/main/ModelNet-C/Train-PS-Point-BERT-PG-ModelNet-C-add_FFM-Augmentation%253AWOLFMix-train_alpha%253A0.5_epoch_300.pt) |
| `ModelNet-C` / `ModelNet40-C` | `Point-MAE` | PSFT | None | [Link](https://huggingface.co/SZUChangMa/PSFT/blob/main/ModelNet-C/Train-PS-Point-MAE-PG-ModelNet-C-add_FFM-Augmentation%253ANone-train_alpha%253A0.5_epoch_300.pt) |
| `ModelNet-C` / `ModelNet40-C` | `Point-MAE` | PSFT + Aug. | WOLFMix | [Link](https://huggingface.co/SZUChangMa/PSFT/blob/main/ModelNet-C/Train-PS-Point-MAE-PG-ModelNet-C-add_FFM-Augmentation%253AWOLFMix-train_alpha%253A0.5_epoch_300.pt) |
| `ModelNet-C` / `ModelNet40-C` | `ULIP-2` | PSFT | None | [Link](https://huggingface.co/SZUChangMa/PSFT/blob/main/ModelNet-C/Train-PS-ULIP-2-PG-ModelNet-C-add_FFM-Augmentation%253ANone-train_alpha%253A0.5_epoch_300.pt) |
| `ModelNet-C` / `ModelNet40-C` | `ULIP-2` | PSFT + Aug. | WOLFMix | [Link](https://huggingface.co/SZUChangMa/PSFT/blob/main/ModelNet-C/Train-PS-ULIP-2-PG-ModelNet-C-add_FFM-Augmentation%253AWOLFMix-train_alpha%253A0.5_epoch_300.pt) |
| `ModelNet-C` / `ModelNet40-C` | `Uni3d-B` | PSFT | None | [Link](https://huggingface.co/SZUChangMa/PSFT/blob/main/ModelNet-C/Train-PS-Uni3d-B-PG-ModelNet-C-add_FFM-Augmentation%253ANone-train_alpha%253A0.5_epoch_300.pt) |
| `ModelNet-C` / `ModelNet40-C` | `Uni3d-B` | PSFT + Aug. | WOLFMix | [Link](https://huggingface.co/SZUChangMa/PSFT/blob/main/ModelNet-C/Train-PS-Uni3d-B-PG-ModelNet-C-add_FFM-Augmentation%253AWOLFMix-train_alpha%253A0.5_epoch_300.pt) |
| `ScanObjectNN-C` | `Point-BERT` | PSFT | None | [Link](https://huggingface.co/SZUChangMa/PSFT/blob/main/ScanObjectNN-C/Train-PS-Point-BERT-PG-ScanObjectNN-C-add_FFM-Augmentation%253ANone-train_alpha%253A0.5_epoch_300.pt) |
| `ScanObjectNN-C` | `Point-BERT` | PSFT + Aug. | WOLFMix | [Link](https://huggingface.co/SZUChangMa/PSFT/blob/main/ScanObjectNN-C/Train-PS-Point-BERT-PG-ScanObjectNN-C-add_FFM-Augmentation%253AWOLFMix-train_alpha%253A0.5_epoch_300.pt) |
| `ScanObjectNN-C` | `Point-MAE` | PSFT | None | [Link](https://huggingface.co/SZUChangMa/PSFT/blob/main/ScanObjectNN-C/Train-PS-Point-MAE-PG-ScanObjectNN-C-add_FFM-Augmentation%253ANone-train_alpha%253A0.5_epoch_300.pt) |
| `ScanObjectNN-C` | `Point-MAE` | PSFT + Aug. | WOLFMix | [Link](https://huggingface.co/SZUChangMa/PSFT/blob/main/ScanObjectNN-C/Train-PS-Point-MAE-PG-ScanObjectNN-C-add_FFM-Augmentation%253AWOLFMix-train_alpha%253A0.5_epoch_300.pt) |
| `ScanObjectNN-C` | `ULIP-2` | PSFT | None | [Link](https://huggingface.co/SZUChangMa/PSFT/blob/main/ScanObjectNN-C/Train-PS-ULIP-2-PG-ScanObjectNN-C-add_FFM-Augmentation%253ANone-train_alpha%253A0.5_epoch_300.pt) |
| `ScanObjectNN-C` | `ULIP-2` | PSFT + Aug. | WOLFMix | [Link](https://huggingface.co/SZUChangMa/PSFT/blob/main/ScanObjectNN-C/Train-PS-ULIP-2-PG-ScanObjectNN-C-add_FFM-Augmentation%253AWOLFMix-train_alpha%253A0.5_epoch_300.pt) |
| `ScanObjectNN-C` | `Uni3d-B` | PSFT | None | [Link](https://huggingface.co/SZUChangMa/PSFT/blob/main/ScanObjectNN-C/Train-PS-Uni3d-B-PG-ScanObjectNN-C-add_FFM-Augmentation%253ANone-train_alpha%253A0.5_epoch_300.pt) |
| `ScanObjectNN-C` | `Uni3d-B` | PSFT + Aug. | WOLFMix | [Link](https://huggingface.co/SZUChangMa/PSFT/blob/main/ScanObjectNN-C/Train-PS-Uni3d-B-PG-ScanObjectNN-C-add_FFM-Augmentation%253AWOLFMix-train_alpha%253A0.5_epoch_300.pt) |

You can download all released checkpoints with:

```bash
pip install huggingface_hub
huggingface-cli download SZUChangMa/PSFT \
  --local-dir ckpts/PSFT \
  --include "ModelNet-C/*.pt" "ScanObjectNN-C/*.pt"
```

## Datasets

Download the clean training data and corrupted test data from Hugging Face.

| Dataset | Classes | Training data | Corrupted test data | Checkpoint |
| :--- | :--- | :--- | :--- | :--- |
| `ModelNet-C` | 40 | [Download](https://huggingface.co/datasets/SZUChangMa/PointCloudCorruption/tree/main/modelnet40_ply_hdf5_2048) | [Download](https://huggingface.co/datasets/SZUChangMa/PointCloudCorruption/tree/main/modelnet_c) | `ModelNet-C` ckpt |
| `ModelNet40-C` | 40 | [Download](https://huggingface.co/datasets/SZUChangMa/PointCloudCorruption/tree/main/modelnet40_ply_hdf5_2048) | [Download](https://huggingface.co/datasets/SZUChangMa/PointCloudCorruption/tree/main/modelnet40_c) | reuse `ModelNet-C` ckpt |
| `ScanObjectNN-C` | 15 | [Download](https://huggingface.co/datasets/SZUChangMa/PointCloudCorruption/tree/main/ScanObjectNN/h5_files/main_split) | [Download](https://huggingface.co/datasets/SZUChangMa/PointCloudCorruption/tree/main/scanobjectnn_c) | `ScanObjectNN-C` ckpt |

## Quick Start

First, clone this repository and install the dependencies:

```bash
git clone <this-repo-url>
cd PSFT-master

pip install torch torchvision torchaudio
pip install numpy h5py scikit-learn timm pyyaml huggingface_hub
pip install "git+git://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
pip install utils/KNN_CUDA-0.2-py3-none-any.whl
```

Download the original backbone checkpoints required by `utils/model.py` and place them under `data_inputs/pretrained/`:

| Backbone | Pretrained checkpoint |
| :--- | :--- |
| `Point-BERT` | [Point-BERT.pth](https://huggingface.co/SZUChangMa/PSFT/blob/main/pretrained/Point-BERT.pth) |
| `Point-MAE` | [Point-MAE.pth](https://huggingface.co/SZUChangMa/PSFT/blob/main/pretrained/Point-MAE.pth) |
| `ULIP-2` | [pretrained_models_ckpt_zero-shot_classification_pointbert_ULIP-2.pt](https://huggingface.co/SZUChangMa/PSFT/blob/main/pretrained/pretrained_models_ckpt_zero-shot_classification_pointbert_ULIP-2.pt) |
| `Uni3d-B` | [Pretrianed_Uni3d_B_Ensembled.pt](https://huggingface.co/SZUChangMa/PSFT/blob/main/pretrained/Pretrianed_Uni3d_B_Ensembled.pt) |

Then evaluate a PSFT checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python test_ps.py \
  --model_path ckpts/PSFT/ModelNet-C/Train-PS-Point-BERT-PG-ModelNet-C-add_FFM-Augmentation%3ANone-train_alpha%3A0.5_epoch_300.pt \
  --model_name Point-BERT \
  --train_mode PG \
  --add_FFM \
  ModelNet-C
```

Replace `--model_path` with the actual downloaded checkpoint path, replace `--model_name` with the matching backbone in the checkpoint name, and replace the final dataset argument with one of:

```text
ModelNet-C
ModelNet40-C
ScanObjectNN-C
```

For `ModelNet40-C`, keep using a checkpoint from the `ModelNet-C` folder and only change the final dataset argument:

```bash
CUDA_VISIBLE_DEVICES=0 python test_ps.py \
  --model_path ckpts/PSFT/ModelNet-C/Train-PS-Point-BERT-PG-ModelNet-C-add_FFM-Augmentation%3ANone-train_alpha%3A0.5_epoch_300.pt \
  --model_name Point-BERT \
  --train_mode PG \
  --add_FFM \
  ModelNet40-C
```

If your data is stored elsewhere, pass dataset-specific paths after the dataset name:

```bash
CUDA_VISIBLE_DEVICES=0 python test_ps.py \
  --model_path ckpts/PSFT/ScanObjectNN-C/Train-PS-Uni3d-B-PG-ScanObjectNN-C-add_FFM-Augmentation%3AWOLFMix-train_alpha%3A0.5_epoch_300.pt \
  --model_name Uni3d-B \
  --train_mode PG \
  --add_FFM \
  ScanObjectNN-C \
  --test_dir /path/to/scanobjectnn_c
```

## Training

PSFT training uses distributed training through `torchrun`. For example:

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 train_ps.py \
  --model_name Point-BERT \
  --train_mode PG \
  --add_FFM \
  ModelNet-C
```

Add WOLFMix augmentation with:

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 train_ps.py \
  --model_name Point-BERT \
  --train_mode PG \
  --add_FFM \
  --add_WOLFMix \
  ModelNet-C
```

## License

The released checkpoints are distributed under the MIT license on [Hugging Face](https://huggingface.co/SZUChangMa/PSFT). Please also check the license terms of the original datasets and backbone checkpoints before use.

## Citation

If you find this work useful, please cite the paper:

```bibtex

@misc{arxiv2607.19711,
  title        = {Point-Selection Fine-Tuning Framework for Robust Point Cloud Classification},
  author       = {Da Li, Chang Ma, and Dongfu Yin},
  year         = {2026},
  url          = {https://arxiv.org/abs/2607.19711}
}
