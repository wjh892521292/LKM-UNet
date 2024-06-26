# LKM-UNet: Large Kernel Vision Mamba for Medical Segmentation elevates SSMs beyond Convolution and Self-attention 🚀




<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2404.02905-b31b1b.svg)](https://arxiv.org/abs/2403.07332)
</div>

<p align="center" style="font-size: larger;">
  <a href="https://arxiv.org/abs/2403.07332">Large Kernel Vision Mamba UNet for Medical Image Segmentation</a>
</p>

![w:800px](imgs/overview2.png)



## Requirements: 

python 3.10 + torch 2.0.1 + torchvision 0.15.2 (cuda 11.8)

If cuda is 11.8 run:
```bash
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html
```

Install [Mamba](https://github.com/state-spaces/mamba): `pip install causal-conv1d` and `pip install mamba-ssm`

Install [monai](https://github.com/Project-MONAI/MONAI): `pip install monai` 

Download code: `git clone https://github.com/wjh892521292/LKM-UNet` and `cd LKM-UNet/lkmunet` and run `pip install -e .`




## Preprocessing

```bash
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```

## Train models

- Train 2D `LKMUNet` model

```bash
nnUNetv2_train DATASET_ID 2d all -tr nnUNetTrainerLKMUNet
```




- Train 3D `LKMUNet` model

```bash
nnUNetv2_train DATASET_ID 3d_fullres all -tr nnUNetTrainerLKMUNet
```




## Inference

- Predict testing cases with `LKMUNet` model

```bash
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_ID -c CONFIGURATION -f 'all' -tr nnUNetTrainerLKMUNet --disable_tta -npp 1
```

> `CONFIGURATION` can be `2d` and `3d_fullres` for 2D and 3D models, respectively.

## Paper

```
@article{wang2024large,
    title={Large Kernel Vision Mamba UNet for Medical Image Segmentation},
    author={Jinhong Wang and Jintai Chen and Danny Chen and Jian Wu},
    journal={arXiv preprint arXiv:2403.07332},
    year={2024}
}
```


## Acknowledgements

Thank the authors of [nnU-Net](https://github.com/MIC-DKFZ/nnUNet), [Mamba](https://github.com/state-spaces/mamba) and [U-mamba](https://github.com/bowang-lab/U-Mamba) for making their valuable code publicly available.
