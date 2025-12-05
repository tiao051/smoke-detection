# Vision-Based Smoke Detection System (YOLOv11)

A state-of-the-art Data-Centric AI approach to automated smoke and fire detection designed to minimize false alarms caused by environmental phenomena like fog, clouds, and water vapor.

---

## Overview

This project addresses a critical challenge in automated fire detection systems: reducing false alarm rates from confusion between smoke and environmental phenomena (fog, mist, clouds, water vapor).

Rather than relying solely on advanced architectures, we implement a hybrid data strategy combining multiple high-quality datasets to create a robust, production-ready smoke detection model.

---

## Key Objectives

- YOLOv11 Architecture: State-of-the-art object detection optimized for speed and accuracy
- Hybrid Data Strategy: Multi-source dataset fusion for improved robustness
- Data-Centric Approach: Focus on data quality and diversity over model complexity
- Production Ready: Export capabilities for real-world deployment
- High Noise Resistance: Distinguish smoke from fog, clouds, and industrial steam

---

## Technology Stack

### Model Architecture
- YOLOv11 (Ultralytics) with flexible deployment options
- Real-time inference optimization for edge deployment
- Multi-format export support (ONNX, TensorRT, CoreML)

### Data Sources
- D-Fire Dataset: Approximately 21,000 images with smoke, fire, and background samples
- FASDD (Fire and Smoke Detection Dataset): Supplementary data containing hard negatives (fog, mist, low clouds)

### Data Strategy
The hybrid data approach combines:
- Core training data from D-Fire for learning fundamental smoke/fire patterns
- Hard negative samples from FASDD to reduce false positives
- Intelligent data splitting: 70% Training, 20% Validation, 10% Testing
- Data validation and quality assurance pipeline

---

## Data-Centric Approach

Unlike traditional model-centric approaches, we prioritize:

**Data Quality**
- Strict validation pipeline
- Corruption detection
- Outlier identification

**Data Diversity**
- Multiple sources (D-Fire, FASDD)
- Various environmental conditions
- Hard negative examples (fog, mist, clouds)

**Data Balance**
- Representative class distribution
- Adequate background samples
- Proper train/val/test split

**Data Consistency**
- Standardized labeling format
- Coordinate normalization
- Uniform image handling

---

## Development Roadmap

- Dataset preparation and validation
- YOLOv11 model training pipeline
- Model evaluation and performance metrics
- Export to production formats (ONNX, TensorRT)
- Real-time inference implementation
- API server development for deployment
- Web dashboard for monitoring and analysis

---

## Credits

### Datasets
- D-Fire Dataset by grizzlynyo
- FASDD (Fire and Smoke Detection Dataset) by FASDD Team

### Framework
- YOLOv11 by Ultralytics
