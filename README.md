# Space Station Object Detection - BuildwithDelhi 2.0 Hackathon

## 🎯 Project Overview
Robust object detection model for space station objects (FireExtinguisher, ToolBox, OxygenTank) using YOLOv8 and synthetic data from Duality AI's Falcon platform.

## 🏆 Results
- **mAP@0.5**: 91.5%
- **mAP@0.5-0.95**: 83.8%
- **Precision**: 96.9%
- **Recall**: 85.4%
- **Speed**: 4.3ms per image (~233 FPS)

## 📊 Per-Class Performance
| Class | mAP@0.5 | mAP@0.5-0.95 |
|-------|---------|--------------|
| FireExtinguisher | 94.8% | 88.1% |
| ToolBox | 91.1% | 85.8% |
| OxygenTank | 88.7% | 77.4% |

## 🚀 Quick Start
pip install -r requirements.txt
python scripts/train.py --epochs 25
python scripts/predict.py --weights weights/best.pt --source <test_images>

## 📁 Repository Structure
- `scripts/`: Training and inference code
- `weights/`: Trained model weights
- `configs/`: Configuration files
- `results/`: Training metrics and visualizations
- `docs/`: Detailed documentation

## 🛠️ Model Architecture
- **Base Model**: YOLOv8n
- **Optimizer**: SGD with momentum 0.9
- **Learning Rate**: 0.0005
- **Augmentation**: Mosaic (0.5), blur, median blur
- **Epochs**: 25

## 📈 Training Process
1. Initial baseline: 73.7% mAP@0.5
2. Hyperparameter optimization
3. Advanced augmentation techniques
4. Final result: 91.5% mAP@0.5

## 🎪 Demo
See `results/validation_predictions/` for sample detection outputs.

## 👥 Team
[Cosmic Crushers]
- [Abhishek Mittal]
- [Shubham Yadav]
- [Krish Mangla]
- [Tavishi.]
- [Lakshay]
