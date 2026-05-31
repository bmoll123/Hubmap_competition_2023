## Introduction

### 1. Dataset Status
* **Task:** Instance segmentation of microvascular structures in human kidney tissue sections.
* **Challenges:** The vessels are densely packed with highly variable morphologies. Crucially, the dataset contains **severe label noise** (only a small subset is clean and expert-annotated, while the majority consists of automatically generated, noisy/imperfect labels).

### 2. Core Architectures
We ensembled three top-tier computer vision models to ensure diverse feature extraction and maximum robustness:
* **ViT-Adapter + HTC++** 
* **DetectoRS (ResNet50 + HTC)** 
* **CBNetv2 + HTC** 

### 3. Three Core Strategies
* **Diverse Strong Baselines:** By leveraging three distinct, state-of-the-art architectures, the pipeline ensures robust coverage of microvessels with irregular and complex shapes.
* **Two-Stage Training for Noisy Labels (Noise Tolerance):** Models are initially pre-trained on the combined dataset (noisy + clean data) at a large scale. In the second stage, they are fine-tuned **exclusively on the high-quality, expert-annotated data** to prevent the models from overfitting to label errors.
* **Custom Mask-Aware Ensemble:** We utilize the **Weighted Boxes Fusion (WBF)** algorithm to fuse predicted bounding boxes, followed by a pixel-level summation of the predicted masks from all three models, strictly retaining the precise microvascular regions within the fused boundaries.


## Environment Setup
```
conda create -n python3.9 python=3.9 -y
conda activate python3.9

pip install -r requirements.txt

pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html
pip install mmcv-full==1.7.2 -f https://download.openmmlab.com/mmcv/dist/cu117/torch2.0.0/index.html
pip install mmdet==2.28.2 mmpretrain==1.0.0rc8

```



## Data preparation
Dataset download：https://github.com/Nischaydnk/HubMap-2023-3rd-Place-Solution/tree/main
```
kaggle datasets download -d nischaydnk/hubmap-coco-datasets

kaggle datasets download -d nischaydnk/hubmap-coco-pretrained-models

kaggle competitions download -c hubmap-hacking-the-human-vasculature
```


```
HubMap-2023-3rd-Place-Solution/
├── hubmap-coco-pretrained-models/
├── hubmap-hacking-the-human-vasculature/
│   ├── coco_data/                        # Extracted dataset from hubmap-coco-datasets.zip
│   ├── polygons.jsonl
│   ├── sample_submission.csv
│   ├── semantic_masks/
│   ├── test/
│   ├── tile_meta.csv
│   ├── train/
│   └── wsi_meta.csv
├── requirements.txt
└── README.md

```

## Training

### Stage 1
```
# vitadapter + htc++
python train.py \
    ./all_configs/pretconf/pretexp1_adaplargebeitv2l_htc-Copy1.py \
    --launcher none \
    --seed 69

# DetectoRS
python train.py \
    ./all_configs/pseudo_config_pret/ds2allwsiprethtc50ps60.py \
    --launcher none \
    --seed 69

# CBNetV2
python train.py \
    ./all_configs/nops_config_pret/pretexp1_cbnetbase_1600_10e.py \
    --launcher none \
    --seed 69
```


### Stage 2
```
# vitadapter + htc++
python train.py \
    ./all_configs/nops_config_finetune/exp4_adapbeitv2l.py \
    --launcher none \
    --seed 69

# DetectoRS
python train.py \
    ./all_configs/pseudo_config_finetune/ds1pretexp1moreaug-htc50-2048-cv408ps.py \
    --launcher none \
    --seed 69

# CBNetV2
python train.py \
    ./all_configs/nops_config_finetune/exp1_withpret_cbbase_1600_morepretep.py \
    --launcher none \
    --seed 69
```

## Kaggle Inference


https://www.kaggle.com/code/yuyuntt/cv-wala-mega-ensemble-hubmap-2023



## Performance on Hubmap Competition

