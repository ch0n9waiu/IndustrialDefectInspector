# IndustrialDefectInspector

An industrial computer vision project for surface defect inspection, covering
dataset auditing, defect detection, error analysis, and deployment-oriented
model engineering.

The project is built around the NEU-DET steel surface defect dataset and is
developed incrementally from dataset understanding to deployable inference.

> Status: Phase 1 completed. Phase 2 — object detection baseline — in progress.

---

## Project Goals

The project focuses on building an end-to-end industrial visual inspection
workflow rather than only training a deep-learning model.

Planned pipeline:

```text
Dataset inspection
    ↓
Object detection
    ↓
Model evaluation and error analysis
    ↓
Targeted data augmentation
    ↓
ONNX Runtime deployment
    ↓
C++ / OpenCV inference
    ↓
TensorRT acceleration
    ↓
Industrial inspection demo
