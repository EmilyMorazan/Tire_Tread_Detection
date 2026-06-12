# 🚗 Tire Tread Depth Detection

An AI-powered computer vision system that estimates tire tread depth from a single smartphone photo.

This project uses a fine-tuned ResNet-18 deep learning model to analyze tire images and predict tread depth as a continuous value, helping drivers assess tire wear without specialized tools.

---

## Overview

Tire tread depth is a critical safety metric that affects:

- Braking performance
- Road traction
- Hydroplaning resistance
- Vehicle safety inspections

Traditional tread measurement requires a physical gauge, while common methods such as the penny or quarter test provide only rough approximations.

This project demonstrates how computer vision and machine learning can transform a simple tire photo into an accurate tread depth estimate.

---

## Features

-Upload a tire image
-AI-powered tread depth prediction
-Continuous regression output (not limited to categories)
-Interactive React + Vite web interface
-Deep learning model trained on real-world tire data
-Fast inference from a single photo

---

## Machine Learning Pipeline

```text
Tire Photo
     ↓
Image Preprocessing
     ↓
ResNet-18 Backbone
(Transfer Learning)
     ↓
Regression Head
     ↓
Predicted Tread Depth
```

The model leverages transfer learning from ImageNet and replaces the final classification layer with a regression head that predicts tread depth directly.

---

## Model Architecture

- Backbone: ResNet-18
- Framework: PyTorch
- Learning Method: Transfer Learning
- Loss Function: Mean Squared Error (MSE)
- Optimizer: Adam
- Input Size: 224 × 224 RGB
- Output: Continuous tread depth estimate

Why regression?

Unlike classification, regression allows the model to predict values such as:

```text
5.3 / 32"
7.8 / 32"
```

instead of placing tires into broad categories.

---

## Dataset

The dataset was collected from tire shops in the Greater Los Angeles area and includes manually measured ground-truth tread depths.

### Dataset Statistics

| Metric | Value |
|----------|---------|
| Total Images | 660 |
| Tread Depth Range | 2/32" – 10/32" |
| Classes | 9 |
| Capture Device | iPhone |
| Training Split | 80% |
| Validation Split | 20% |

Each tire was measured using a tread gauge before imaging to ensure accurate labels. :contentReference[oaicite:1]{index=1}

---

## Results

### Final Performance

| Metric | Value |
|----------|---------|
| Mean Absolute Error (MAE) | 0.2573 |
| Best Validation Loss | 0.1948 |
| Total Training Epochs | 155 |

The model achieved an average prediction error of approximately:

```text
0.26 / 32"
```

demonstrating strong performance despite a relatively small dataset. :contentReference[oaicite:2]{index=2}

---

## Technologies Used

### Frontend

- React
- Vite
- JavaScript
- CSS

### Machine Learning

- Python
- PyTorch
- Torchvision
- NumPy

### Computer Vision

- Image preprocessing
- Data augmentation
- Transfer learning

---

## Project Structure

```text
tire-tread-demo/
│
├── public/
├── src/
│   ├── assets/
│   ├── App.jsx
│   ├── TireTreadDemo.jsx
│   ├── main.jsx
│   └── styles
│
├── package.json
├── vite.config.js
└── README.md
```

---

## Future Improvements

- Larger and more balanced dataset
- Detection of uneven tire wear
- Multi-camera and multi-lighting robustness
- Mobile deployment (iOS / Android)
- Real-time camera-based tread analysis
- Safety recommendations and replacement alerts

---

## Research Impact

This project explores how deep learning can make vehicle safety assessments more accessible and affordable.

By eliminating the need for specialized measuring tools, AI-based tread analysis has the potential to:

- Increase driver awareness
- Improve preventive maintenance
- Reduce tire-related accidents
- Enable mobile-first safety inspections

---

## Authors

Spencer Caillat  
Emily Morazan

California State University, Northridge (CSUN)

---

## License

This project is intended for educational and research purposes.
