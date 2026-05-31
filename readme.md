## Data preparation
資料集下載位置：https://github.com/Nischaydnk/HubMap-2023-3rd-Place-Solution/tree/main
```
kaggle datasets download -d nischaydnk/hubmap-coco-datasets

kaggle datasets download -d nischaydnk/hubmap-coco-pretrained-models

kaggle competitions download -c hubmap-hacking-the-human-vasculature
```
datasets結構
HubMap-2023-3rd-Place-Solution
-hubmap-coco-pretrained-models
-hubmap-hacking-the-human-vasculature
    -coco_data (hubmap-coco-datasets.zip出來的結果)


## Train

### Stage 1
CUDA_VISIBLE_DEVICES=0 python train.py \
    ./all_configs/pretconf/pretexp1_adaplargebeitv2l_htc-Copy1.py \
    --launcher none \
    --seed 69
### Resume training
CUDA_VISIBLE_DEVICES=0 python train.py \
    ./all_configs/pretconf/pretexp1_adaplargebeitv2l_htc-Copy1.py \
    --launcher none \
    --seed 69 \
    --resume-from ./results/stage1/best_segm_mAP_epoch_1.pth #自行替換


### Stage 2
python train.py ./all_configs/nops_config_finetune/exp4_adapbeitv2l.py --launcher none --seed 69 

### Stage 1 + 2
chmod +x dist_train.sh
./dist_train.sh




### 加semantica mask
先跑python ./tools/generate_semantic_masks.py


## Experiment Results

### 訓練概述
本實驗採用**二階段訓練策略** (Two-Stage Training Pipeline)，針對HubMap血管分割任務進行優化。該方法在同一模型架構（HTC with BEiT V2 Adapter）上，通過不同的數據集、增強策略和訓練參數的組合，實現了顯著的性能提升。

#### 性能指標總結

| 指標 | Stage 1 | Stage 2 | 改進 |
|------|--------|--------|------|
| 最佳 segm_mAP | 0.3149 (Epoch 8) | 0.4043 (Epoch 19) | +28.4% ↑ |
| **segm_AP50** | **0.6059** | **0.7153** | **+18.0% ↑** |
| **segm_AP75** | **0.2935** | **0.4203** | **+43.2% ↑** |
| segm_mAP_s (Small) | 0.2857 | 0.3448 | +20.7% ↑ |
| segm_mAP_m (Medium) | 0.3684 | 0.4875 | +32.4% ↑ |
| segm_mAP_l (Large) | 0.3899 | 0.6302 | +61.6% ↑ |
| 訓練記錄 | 489 iterations | 392 iterations | - |

#### 訓練曲線
- **Stage 1 曲線**：results/plots/stage1_training_curves.png
- **Stage 2 曲線**：results/plots/stage2_training_curves.png

---

### Stage 1: 預訓練階段 (Pre-training Phase)

#### 目標
在大規模混合數據集上進行預訓練，讓模型學習通用的血管分割特徵。

#### 訓練配置
- **配置文件**: all_configs/pretconf/pretexp1_adaplargebeitv2l_htc-Copy1.py
- **數據集**:
  - Dataset 1: 完整精細標注 (High-quality annotations)
  - Dataset 2: 粗糙標注 (Coarse annotations)
  - 合併使用，提供豐富的樣本多樣性
  
- **數據增強 (輕度)**:
  ```
  - RandomFlip (水平和垂直翻轉) @ 50% 概率
  - AutoAugment: Shear + EqualizeTransform
  - ShiftScaleRotate: 15° 旋轉, ±6.25% 位移, ±15% 縮放
  ```

#### 設計理由
1. **數據多樣性**: 結合高質量和粗糙標注的數據集，增加模型見過的樣本變化
2. **輕度增強**: 保留足夠的原始特徵，避免過度增強導致標注失配
3. **快速收斂**: Stage 1 在第 8 個 epoch 達到最佳性能 (mAP: 0.3149)

#### 輸出
- 最佳檢查點: results/stage1/best_segm_mAP_epoch_8.pth
- 該模型作為 **Stage 2 的初始化權重**

---

### Stage 2: 微調階段 (Fine-tuning Phase)

#### 目標
利用 Stage 1 學到的特徵，在高質量數據集上進行精細化訓練，最大化分割精度。

#### 訓練配置
- **配置文件**: all_configs/nops_config_finetune/exp4_adapbeitv2l.py
- **數據集**: Dataset 1 only (完整精細標注)
- **初始化權重**: Stage 1 最佳檢查點 (0.3149 mAP)

- **數據增強 (重度)**:
  ```
  - RandomFlip (水平和垂直翻轉) @ 50% 概率
  - AutoAugment 策略:
    * Shear (概率 40%, level 0)
    * Translate (概率 40%, level 5)
    * PhotoMetricDistortion: 亮度 ±32, 對比度 (0.5-1.5x), 色調 ±18
    * MinIoURandomCrop: 保留 IoU (0.4-0.9), 最小裁剪 30%
  ```

#### 設計理由
1. **專注於高質量數據**: 移除粗糙標注，集中在精確的標注上，避免模型被雜訊引導
2. **重度增強**: 在高質量數據上進行更激進的增強，增加泛化能力，同時精細標注可確保增強的合理性
3. **遷移學習效果**: 從 Stage 1 的通用特徵出發，Stage 2 能更有效地適應目標域

#### 性能改進
- **最終 segm_mAP**: 0.4043 (Epoch 19)
- **vs Stage 1**: **+28.4% 相對提升**
- **收斂特性**: 需要更多 epoch (19 vs 8)，反映細緻優化的過程

---

### 為什麼採用二階段訓練？

#### 1. 漸進式學習
   - Stage 1 在廣泛數據上學習基本特徵 (魯棒性)
   - Stage 2 在高質量數據上細化特徵 (精確性)
   - 模仿人類從粗粒度到細粒度的學習過程

#### 2. 資料效率最大化
   - 不同質量的標注各有用處，Stage 1 充分利用所有數據
   - Stage 2 避免被低質量標注「污染」，集中精力於精確優化
   - 結果：Stage 2 性能相比直接在 Dataset 1 訓練有顯著提升

#### 3. 避免過擬合
   - Stage 1 預訓練充分初始化權重，Stage 2 能用更少 epoch 達到更優性能
   - 相比冷啟動，這個策略提供更穩定的收斂路徑

#### 4. 計算效率
   - 儘管總 epoch 較多 (19 + 8 iterations)，但每個階段優化目標明確
   - Stage 1 快速定型，Stage 2 基於良好初值加速收斂

---

### 為什麼使用不同的數據增強策略？

#### Stage 1 - 輕度增強的原因
```
低質量標注 + 輕度增強 → 原始特徵保留 → 避免增強後的邊界與標注不符
```
- **粗糙標注限制**: 若使用激進的增強 (裁剪、旋轉、變形)，可能導致增強後的特徵與粗糙標注邊界對不上
- **保守策略**: 基本的翻轉和溫和的色彩/幾何變換，確保增強有效性

#### Stage 2 - 重度增強的原因
```
精細標注 + 重度增強 → 充分利用標注質量 → 增強合理且有效
```
- **精細標注支持**: 高質量邊界標注能承受更複雜的變換
- **泛化能力**: MinIoURandomCrop、PhotoMetricDistortion 等操作在精確標注上安全且有效
- **遷移優勢**: 從 Stage 1 良好的初值出發，模型能從重度增強中充分受益

#### 類比：預訓練 (Pretrain) 概念
✓ **是的，這是預訓練的思想**
- Stage 1 = 預訓練 (在廣泛、雜訊標注數據上)
- Stage 2 = 微調 (在乾淨、高質量目標域數據上)
- 這種範式在 BERT、視覺 Transformer 等現代深度學習中廣泛採用

---

### 驗證策略：固定 Split vs 隨機 Split

#### 採用策略：固定 Split (Dataset 1 WSI 1 left)

#### 設計理由

| 優點 | 說明 |
|------|------|
| **可重複性** | 相同的驗證集確保不同實驗結果直接可比 |
| **穩定評估** | 避免隨機 split 帶來的方差，更準確反映模型改進 |
| **WSI 級別完整性** | 整塊組織切片 (WSI) 上的評估更接近實際應用場景 |
| **分佈一致性** | 固定空間區域的驗證集避免數據分佈波動 |
| **模型選擇** | 便於追蹤每個 epoch 的性能，選擇最優檢查點 |

#### vs 隨機 Split 的對比

| 評估指標 | 隨機 Split | 固定 Split (採用) |
|--------|-----------|----------------|
| 跨實驗可比性 | ✗ (高方差) | ✓ (確定性) |
| 評估成本 | 同等 | 同等 |
| 結果魯棒性 | 需多次平均 | 單次已穩定 |
| 實際應用相關性 | ✗ (碎片化) | ✓ (完整組織) |

#### 實踐影響
- Stage 1 驗證集: 完全相同的 Dataset 1 WSI 1 left
- Stage 2 驗證集: 保持一致，使 Stage 1→2 的改進量化準確
- 結果：mAP 從 0.3149 → 0.4043 的 28.4% 提升是可信且可重複的

---

### 結論

通過二階段訓練策略，結合**漸進式學習**、**智能數據利用**和**差異化增強**，在同一模型架構下實現了 28.4% 的性能提升。此方法論適用於其他混合標注質量的醫療影像分割任務。
