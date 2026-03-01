
<div align="center">

<img src="https://cdn-icons-png.flaticon.com/512/4712/4712100.png" width="90" height="90" alt="YOLO Logo">

# 🎯 YOLO Object Detection

<h3 align="center">
  <img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&weight=600&size=26&pause=1000&color=3B82F6&center=true&vCenter=true&width=700&lines=Real-Time+Object+Detection;Deep+Learning+Computer+Vision;YOLOv8+End-to-End+Pipeline" alt="Typing SVG" />
</h3>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB.svg?style=for-the-badge&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=for-the-badge&logo=pytorch&logoColor=white">
  <img src="https://img.shields.io/badge/Ultralytics-YOLOv8-111111.svg?style=for-the-badge">
  <img src="https://img.shields.io/badge/Computer--Vision-AI-blueviolet?style=for-the-badge">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/mAP@0.5-0.98-success?style=for-the-badge">
  <img src="https://img.shields.io/badge/mAP@0.5:0.95-0.62-blue?style=for-the-badge">
</p>

</div>

---

## 🌟 Overview

End-to-end object detection project using **YOLOv8**.  
Includes dataset preprocessing, training, evaluation, and inference pipeline.

---

## 🎯 Features

- CSV → YOLO annotation conversion  
- Train / Validation split  
- YOLOv8 model training  
- Precision, Recall, mAP evaluation  
- Bounding box visualization  

---

## 🏗️ Pipeline

```mermaid
graph TB
    A[Raw Images] --> B[CSV Annotations]
    B --> C[YOLO Format Conversion]
    C --> D[Model Training]
    D --> E[Evaluation]
    E --> F[Inference]
````

---

## 📊 Model Performance

| Metric       | Score |
| ------------ | ----- |
| Precision    | ~0.95 |
| Recall       | ~0.93 |
| mAP@0.5      | ~0.98 |
| mAP@0.5:0.95 | ~0.62 |

---

## 🧰 Tech Stack

* Python
* PyTorch
* Ultralytics YOLOv8
* OpenCV
* Pandas

---

## 🚀 Installation

```bash
git clone https://github.com/yourusername/yolo-object-detection.git
cd yolo-object-detection
pip install -r requirements.txt
```

---

## ▶ Training

```bash
python src/yolocode.py
```

---

## 🔎 Inference Example

```python
from ultralytics import YOLO

model = YOLO("runs/exp/weights/best.pt")
results = model.predict(source="image.jpg", show=True)
```

---

## 📁 Structure

```
yolo-object-detection/
├── src/
│   └── yolocode.py
├── data/
├── runs/
├── README.md
├── requirements.txt
└── .gitignore
```

---

<p align="center">
  Built with Deep Learning • Computer Vision • YOLOv8
</p>
```
