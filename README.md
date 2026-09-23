# IndustrialDefectInspector

An industrial computer vision project for steel surface defect inspection, covering dataset analysis, object detection, model evaluation, and deployment.

> Current status: **Phase 1 completed — Dataset Understanding & Inspection**  
> Phase 2 in progress — **Object Detection Baseline**

---

## Project Goal

The goal of this project is to build an end-to-end industrial defect inspection pipeline rather than only training a detection model.

Planned workflow:

```text
Dataset inspection
→ Object detection
→ Error analysis
→ Segmentation / anomaly detection
→ ONNX deployment
→ C++ inference
→ TensorRT optimization
→ Industrial inspection demo
```

The project focuses on both:

- computer vision / deep learning algorithms;
- engineering and deployment for industrial applications.

---

## Dataset

This project currently uses the **NEU-DET steel surface defect dataset**.

The dataset contains six defect classes:

- crazing
- inclusion
- patches
- pitted_surface
- rolled-in_scale
- scratches

Dataset summary:

| Item | Count |
|---|---:|
| Images | 1800 |
| XML annotations | 1800 |
| Annotated objects | 4189 |
| Defect classes | 6 |
| Multi-class images | 123 |
| Maximum objects per image | 9 |

The dataset itself is not included in this repository.

Expected local structure:

```text
data/
└── NEU-DET/
    ├── ANNOTATIONS/
    └── IMAGES/
```

---

## Phase 1 — Dataset Understanding & Inspection

Phase 1 focuses on validating the dataset before model training.

Implemented checks include:

- dataset structure inspection;
- XML annotation parsing;
- image/XML pairing validation;
- image readability checks;
- image/XML dimension consistency;
- bounding-box validity and boundary checks;
- class and object statistics;
- single-class vs multi-class analysis;
- whole-image brightness and contrast analysis;
- bounding-box ROI analysis;
- ROI vs whole-image brightness / contrast delta analysis;
- histogram and boxplot visualization.

### Data Integrity Results

The dataset passed the current integrity checks:

- invalid bounding boxes: **0**
- out-of-bounds boxes: **0**
- image/XML filename mismatches: **0**
- images without annotations: **0**
- annotations without images: **0**
- image read errors: **0**
- image/XML size mismatches: **0**

---

## Defect Distribution

Number of annotated objects:

| Defect class | Objects |
|---|---:|
| crazing | 689 |
| inclusion | 1011 |
| patches | 881 |
| pitted_surface | 432 |
| rolled-in_scale | 628 |
| scratches | 548 |

The dataset is not perfectly balanced at the object level, which will be considered during detector evaluation.

---

## Image-Level Analysis

Whole-image brightness and contrast distributions show noticeable differences across defect classes.

Examples:

- `pitted_surface` images tend to have relatively high overall brightness;
- `scratches` images tend to be darker;
- `patches` has relatively high image-level contrast;
- class co-occurrence can noticeably affect image-level statistics.

For this reason, single-class images were also analyzed separately to reduce confounding from multiple defect classes appearing in one image.

---

## ROI Analysis

Bounding-box regions were compared with their corresponding whole images.

For each annotated object:

```text
brightness delta = ROI brightness - whole-image brightness

contrast delta = ROI contrast - whole-image contrast
```

Notable observations include:

| Defect class | Mean brightness delta | Mean contrast delta |
|---|---:|---:|
| crazing | -2.09 | -3.05 |
| inclusion | -4.07 | -4.45 |
| patches | -42.94 | -14.09 |
| pitted_surface | +0.60 | -4.68 |
| rolled-in_scale | -2.11 | -0.16 |
| scratches | +21.33 | +6.56 |

`patches` bounding-box regions are typically substantially darker than their corresponding whole images, while `scratches` regions tend to be brighter.

Boxplot analysis indicates that these shifts are not explained only by a small number of extreme samples.

### Important Limitation

A bounding box contains both defect pixels and surrounding background.

Therefore:

> ROI statistics must not be interpreted as pure defect-pixel statistics.

Similarly, statistical association between defect class and brightness / contrast does not prove that a trained model actually relies on those features.

Model reliance must be tested experimentally after a baseline detector is trained.

---

## Bounding Box Convention

NEU-DET annotations were found to be consistent with a **1-based inclusive** bounding-box coordinate convention.

For NumPy slicing:

```python
roi = image[ymin - 1:ymax, xmin - 1:xmax]
```

For OpenCV visualization:

```python
upper_left = (xmin - 1, ymin - 1)
lower_right = (xmax - 1, ymax - 1)
```

This convention was checked against dataset-wide coordinate bounds and visualized annotations.

---

## Current Repository Structure

```text
IndustrialDefectInspector/
├── scripts/
│   └── inspect_dataset.py
├── README.md
├── .gitignore
└── .gitattributes
```

Local-only directories such as datasets, model weights, outputs, virtual environments, and private development notes are excluded from version control.

---

## Run Phase 1 Inspection

Install the required dependencies:

```bash
python -m pip install opencv-python numpy matplotlib
```

Place NEU-DET under:

```text
data/NEU-DET/
```

Then run:

```bash
python scripts/inspect_dataset.py
```

The script performs dataset validation, statistical analysis, and generates brightness / contrast and ROI-delta boxplots.

---

## Roadmap

### Phase 1 — Dataset Understanding & Inspection

- [x] Dataset integrity validation
- [x] XML and bounding-box inspection
- [x] Class distribution analysis
- [x] Single-class / multi-class analysis
- [x] Brightness and contrast analysis
- [x] ROI-level analysis
- [x] ROI delta analysis

### Phase 2 — Object Detection Baseline

- [ ] Train / validation / test split
- [ ] Convert annotations for detector training
- [ ] Train first YOLO baseline
- [ ] Precision / Recall / mAP evaluation
- [ ] Per-class performance analysis
- [ ] False-positive / false-negative analysis
- [ ] Targeted augmentation experiment

### Phase 3 — Deployment

- [ ] Export model to ONNX
- [ ] ONNX Runtime inference
- [ ] C++ / OpenCV inference pipeline
- [ ] Latency and throughput benchmarking
- [ ] TensorRT FP16 optimization

### Later Extensions

- [ ] Defect segmentation
- [ ] Anomaly detection
- [ ] Traditional machine vision methods
- [ ] Industrial inspection demo

---

## Tech Stack

Currently used / planned:

- Python
- OpenCV
- NumPy
- Matplotlib
- PyTorch
- YOLO
- ONNX Runtime
- C++
- TensorRT

---

## Status

The project is under active development.

**Current stage:** Phase 2 — Object Detection Baseline.
