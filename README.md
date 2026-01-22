# Real-Time-Defect-Detection-Capstone-Project

YOLOv8-based object detection system for identifying common FDM print defects in real time.

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

## Part 1 - Preprocessing
1) Import the dataset to the "data/" file. For the training data it's "train_data" and for testing it's "test_data".
2) Run the data_splitter.py to make train, validation and test dataset splits.
3) Run the FIRST FIVE cells in the training_loop.ipynb to complete the preprocessing part

## Part 2 - Hyperparameter Tuning
1) Run the NEXT THREE cells in the training_loop.ipynb to finish the hyperparameter tuning

## Part 3 - Training and Validation
1) Run the NEXT THREE cells in the training_loop.ipynb to setup the training process, train the baseline model and evaluate it on the validation dataset.
