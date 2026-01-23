# Real-Time-Defect-Detection-Capstone-Project

YOLOv8-based object detection system for identifying common FDM print defects in real time.

---

## Motivation

Fused Filament Fabrication (FFF) in 3D printing is widely used in rapid prototyping and low-volume manufacturing, yet print failures due to defects can lead to material waste, machine downtime, and increased production costs. Manual monitoring is inefficient and error-prone.

This project aims to develop a real-time computer vision system capable of automatically detecting common FDM print defects during the printing process, enabling early intervention and improved print reliability.

---

## System Overview

The overall pipeline of the project is as follows:

1. Images are collected from FFF printing processes which are already made as datasets
2. Defects are annotated using bounding boxes in YOLO format
3. The dataset is split into training, validation, and test sets
4. YOLOv8 models are trained using transfer learning
5. Hyperparameters are tuned via grid search using validation performance
6. The best model is selected and evaluated on a held-out test set
7. The final model is used for real-time inference on GPU hardware

---
## Features

- Real-time defect detection using **YOLOv8**
- Support for multiple FDM defect classes
- GPU-accelerated training and inference
- Modular **training, validation, and testing** pipeline
- Reproducible **hyperparameter grid search**
- Clear separation between model selection and final testing

---

## Dataset

- Images collected from the project dataset drive
- Annotations in **YOLO format**
- Object detection task with bounding boxes

### Defect Classes

- Spaghetti  
- Poor Bridging  
- Overhang Sag  
- Shifted Layer  
- Warping  
- Delamination  
- Foreign Body  
- Stringing  

---

## Project Structure

```text
data/
  train/
    images/
    labels/
  val/
    images/
    labels/
  test/
    images/
    labels/
  data.yaml
runs/
    detect/
        train_final/
        val/
        test/
        test_metrics/
    gridsearch/
    final_gridsearch
scripts/
    data_splitter.py
    training_loop.ipynb
    Change class.py
README.md

---

## Hyperparameter Tuning

A reproducible grid search was conducted to optimize model performance. The following hyperparameters were explored:

- Learning rate
- Weight decay
- Box loss
- Classification loss
- Number of frozen layers

Model selection was based exclusively on validation performance to avoid test set leakage.

---

## Evaluation Metrics

Model performance is evaluated using YOLO-style metrics:

- **mAP@50**
- **mAP@50–95**
- Precision
- Recall
- Per-class Average Precision

All final metrics are reported on a held-out test set that was not used during training or hyperparameter selection.

---

## Environment

The experiments in this project were conducted using the following hardware and software environment.

---

### Hardware

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM)
- CPU: x86_64 architecture
- RAM: 32 GB

---

### Software

- Operating System: Windows 11
- Python: 3.12.x
- PyTorch: 2.5.1 (CUDA-enabled)
- CUDA: 12.1
- Ultralytics YOLOv8: 8.x

---

### Notes

- GPU acceleration is required for efficient training and inference.
- All experiments were run using the Ultralytics Python API.
- Validation and test metrics follow the COCO evaluation protocol.

---

## Project Workflow

### Part 1 – Preprocessing

- Import the dataset into the `data/` directory  
  - Training data: `train_data`
  - Testing data: `test_data`
- Run `data_splitter.py` to generate train, validation, and test splits
- Run the **first six cells** in `training_loop.ipynb` to complete preprocessing

### Part 2 – Hyperparameter Tuning

- Run the **next three cells** in `training_loop.ipynb` to perform hyperparameter tuning

### Part 3 – Training and Validation

- Run the **next three cells** in `training_loop.ipynb` to:
  - Set up the training configuration
  - Train the baseline model
  - Evaluate performance on the validation dataset

### Part 4 – Testing and Evaluation

- Run the remaining cells in `training_loop.ipynb` to:
  - Evaluate the final model on the test set
  - Generate quantitative metrics and qualitative results
- Final metrics are stored in:
  - `runs/detect/test_metrics`
- Predicted images are stored in:
  - `runs/detect/test`

---

## Example Results

Example detection results, including predicted bounding boxes and defect classes, can be found in `runs/detect/test`:

<p align="center">
  <img src="runs/detect/test/0fd8cfb0-left_photo_85_2024-10-09_18-09-58.jpg" width="280">
  <img src="runs/detect/test/6a1e82a6-left_260photo_2024-10-04T11_24_27-418248.jpg" width="280">
  <img src="runs/detect/test/7e58c855-left_photo_359_2024-10-30_12-24-56.jpg" width="280">
</p>

These visualizations demonstrate the model’s ability to localize and classify multiple defect types under varying print conditions.

---

## Acknowledgments

- Ultralytics YOLOv8 framework
- Project supervisors and dataset contributors
