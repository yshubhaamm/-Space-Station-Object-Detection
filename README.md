# 🚀 Space Station Object Detection

This repository contains the complete pipeline for training, evaluating, and running inference on a YOLOv8 model to detect critical space station equipment (🔧 Toolbox, 🛢️ OxygenTank, 🔥 FireExtinguisher) using synthetic data from Duality AI’s Falcon platform.

---

## 📖 Table of Contents
1. [⚙️ Environment Setup](#environment-setup)  
2. [📂 Repository Structure](#repository-structure)  
3. [📊 Data Preparation](#data-preparation)  
4. [🏋️ Training the Model](#training-the-model)  
5. [📈 Evaluating the Model](#evaluating-the-model)  
6. [🔍 Running Inference](#running-inference)  
7. [🔄 Reproducing Final Results](#reproducing-final-results)  
8. [📤 Expected Outputs](#expected-outputs)  
9. [🧠 Interpreting Results](#interpreting-results)  

---

## ⚙️ Environment Setup

1. **Clone the repository**  
git clone https://github.com/yourusername/space-station-object-detection.git
cd space-station-object-detection

2. **Create a virtual environment** (🔧 optional but recommended)  
python3 -m venv venv
source venv/bin/activate # Linux/Mac
venv\Scripts\activate # Windows

3. **Install dependencies**
pip install -r requirements.txt

> 💡 **Note:** Requires Python ≥3.8 and PyTorch with CUDA support for GPU acceleration.

---

## 📂 Repository Structure
```text
├── configs/
│ └──📝 yolo_params.yaml
├── scripts/
│ ├──🏋️ train.py
│ ├──🔍 predict.py
│ └──🎨 visualize.py
├── weights/
│ ├──🎯 best.pt
│ └──🏁 last.pt
├── results/
│ ├──📊 training_results.png
│ ├──🎯 confusion_matrix.png
│ ├──📑 results.csv
│ └──🖼️ validation_predictions/
│ ├── val_batch0_pred.jpg
│ └── …
├──📄 README.md
└──📦 requirements.txt
```

---

## 📊 Data Preparation

1. **Dataset Structure**  
   Place `HackByte_Dataset` in `data/`:
```text  
data/
├── train/
│ ├── images/
│ └── labels/
├── val/
│ ├── images/
│ └── labels/
└── test/
├── images/
└── labels/
```

2. **Configuration**  
Edit `configs/yolo_params.yaml`:
train: data/train
val: data/val
test: data/test
nc: 3
names: ['FireExtinguisher', 'ToolBox', 'OxygenTank']

---

## 🏋️ Training the Model

Run training with YOLOv8:
python scripts/train.py
--data configs/yolo_params.yaml
--epochs 25
--optimizer SGD
--lr0 0.0005
--mosaic 0.5
--imgsz 640

✅ Weights & logs → `runs/detect/train/`  
🎯 `best.pt` → highest validation mAP@0.5

---

## 📈 Evaluating the Model

Validate on test set:
from ultralytics import YOLO
model = YOLO('weights/best.pt')
results = model.val(data='configs/yolo_params.yaml', split='test')
print(results.box.map)
🔧 Or via CLI:
!yolo val model=weights/best.pt data=configs/yolo_params.yaml

---

## 🔍 Running Inference

Detect on custom images:
python scripts/predict.py
--weights weights/best.pt
--source data/test/images
--conf 0.25
--save
🔖 Outputs → `runs/detect/predict/`

---

## 🔄 Reproducing Final Results

1. Ensure paths in `configs/yolo_params.yaml` match your setup.  
2. Use same hyperparameters & code.  
3. Evaluate with `best.pt` to match reported scores:  
   - **mAP@0.5: 0.915**  
   - **mAP@0.5-0.95: 0.838**

---

## 📤 Expected Outputs

- **📊 `training_results.png`** → Loss & metrics plots  
- **🎯 `confusion_matrix.png`** → Class confusion  
- **📑 `results.csv`** → Epoch-wise metrics  
- **🖼️ `validation_predictions/`** → Sample annotated images

---

## 🧠 Interpreting Results

- **mAP@0.5:** Precision @ IoU≥0.5 (target ≥85%)  
- **mAP@0.5-0.95:** Precisions @ varied IoU (target ≥75%)  
- **Precision:** Correct detections / total predicted (target ≥90%)  
- **Recall:** Correct detections / total actual (target ≥80%)  
- **Inference Speed:** Time per image (pre + inf + post)  

> 🚀 High metrics confirm robust, real-time detection capabilities!

---

**© 2025 Team Cosmic Crushers**  
