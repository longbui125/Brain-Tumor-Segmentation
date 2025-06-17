# Brain Tumor Segmentation
A deep learning project for brain tumor segmentation using U-Net architecture and medical imaging datasets.

# Overview
This project applies a U-Net-based Convolutional Neural Network for semantic segmentation of brain tumors in MRI scans.
The model is trained to generate precise binary masks highlighting tumor regions. It supports both training and inference pipelines, with evaluation metrics and visualization.

## Tech Stack
- Python 3.10
- TensorFlow / Keras
- OpenCV
- NumPy
- Albumentations (for image augmentation)
- Matplotlib
- jupyter Notebook (for training/evaluation)

## Project Structure
brain_tumor_segmentation/

```
├── data/
│   ├── train images/       # Raw MRI brain images for training
│   ├── train masks/        # Ground truth masks
│   └── test images/        # Test images for prediction
│
├── model_build.py          # U-Net model definition
├── train_model.py          # Training loop and augmentation
├── test_model.py        # Predict and visualize on test image
├── utils.py                # Helper functions for loading, preprocessing, and metrics
├── evaluate_model.ipynb    # Model evaluation and visualization        
├── README.md
```
