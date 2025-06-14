# DL_hw2
# Unified Multi-Task Learning System

本專案實作一個輕量級的多任務學習系統，基於單一分支架構 (Single-Head)，結合三項電腦視覺任務：

- 物件偵測 (Object Detection, Mini-COCO)
- 語意分割 (Semantic Segmentation, Mini-VOC)
- 影像分類 (Image Classification, Imagenette-160)

並導入 **Replay Buffer** 作為防止災難性遺忘（Catastrophic Forgetting）的策略，支援三階段訓練流程與完整評估。

---

## 專案目標

- 使用 Fast-SCNN 為 backbone，達成 inference time < 150ms
- 使用單一輸出頭（Unified Head）分別支援三任務
- 搭配 Replay Buffer 強化多階段訓練穩定性
- 模型參數量 < 8M，支援 YOLO-style 偵測格式

---

## 模型架構圖

![Model + Training Flow](https://i.imgur.com/KlR2BeK.png)

---

## 🔧 訓練流程說明

1. **Stage 1：Segmentation 訓練**
   - 更新 segmentation head，儲存 replay buffer
2. **Stage 2：Detection 訓練**
   - 訓練偵測 head
   - 同時 replay segmentation buffer
3. **Stage 3：Classification 訓練**
   - 訓練分類 head
   - 同時 replay segmentation + detection buffer

每個階段均進行評估（mIoU / mAP@50 / Top-1 Accuracy）並確認 performance drop ≤ 5%。

---

## 資料集目錄結構
data/
├── imagenette_160/
│ ├── train/
│ └── val/
├── mini_voc_seg/
│ ├── train/
│ └── val/
├── mini_coco_det/
│ ├── train/
│ ├── val/
│ └── annotations/
│ ├── mini_instances_train2017.json
│ └── mini_instances_val2017.json
## 執行後將自動完成以下流程：

 - 建立三種 DataLoader（Seg, Det, Cls）

 - 初始化 Replay Buffer

 - 依序執行三階段訓練並保存最佳模型

 - 畫出訓練曲線（mIoU / mAP / Accuracy）

 - 評估是否符合「性能下降 ≤ 5%」的作業要求
## 結果範例
Stage1 best mIoU: 0.6548
Stage2 best mAP@50: 0.5123
Stage3 best Top1 Acc: 0.7812

→ Final Seg mIoU: 0.6412, Det mAP@50: 0.5057, Cls Acc: 0.7710
✔ 最終模型評估（計算是否低於 base - 5%）

Seg mIoU: baseline=0.6548, final=0.6412, drop=0.0136 → ✅ PASS
Det mAP@50: baseline=0.5123, final=0.5057, drop=0.0066 → ✅ PASS
Cls Top1 Acc: baseline=0.7812, final=0.7710, drop=0.0102 → ✅ PASS

🎉 所有任務性能下降皆在 5% 以內，符合需求！
