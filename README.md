# Deep Learning Model for Image Classification (Planes vs. Cars)

## Overview

This repository contains a deep learning model for binary image classification of planes and cars. The model is built using PyTorch and trained on datasets from Hugging Face. It includes data preprocessing, augmentation, model training, evaluation, and automated uploading to the Hugging Face Model Hub.

## Features

- **Dataset Handling**: Loads and labels datasets from Hugging Face. These can be found in the following repositories:
  - Planes: [fearlixg/planes\_splitted](https://huggingface.co/datasets/fearlixg/planes_splitted)
  - Cars: [fearlixg/cars\_splitted](https://huggingface.co/datasets/fearlixg/cars_splitted)
- **Data Preprocessing**: Merges, balances, and augments datasets for robust training.
- **Deep Learning Model**: Implements a convolutional neural network (CNN) using PyTorch.
- **Training Pipeline**: Includes loss functions, optimizers, and learning rate scheduling.
- **Evaluation Metrics**: Computes accuracy, precision, recall, F1-score, and ROC-AUC.
- **Model Uploading**: Automates Hugging Face model repository management.
- **Colab Integration**: Fully optimized for Google Colab.

## Installation

Before running the code, install the required dependencies:

```bash
pip install datasets torchvision torch huggingface_hub pandas numpy matplotlib seaborn scikit-learn tqdm
```

## Usage

### Running on Google Colab

To execute the pipeline in Google Colab:

1. Download the file from `colab_project/DSTI.ipynb`.
2. Open the [Google Colab ](https://colab.research.google.com)
3. Go to the top left corner and open the notebook.
4. Add your HF_TOKEN under secrets.
5. Run the notebook cell by cell.

### Running Locally with Python Script

A standalone Python script `local_project/DSTI.py` is provided for running the model outside Google Colab. Ensure you have all dependencies installed by using the provided `local_project/requirements.txt` file. Please also add your secret to the secrets.json!!!

#### 1. Clone the Repository

```bash
git clone https://github.com/fearlix/drl.git
cd drl/local_project
```

#### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 3. Run the Model

```bash
python main.py
```

This will:
- Authenticate Hugging Face.
- Load and preprocess datasets.
- Train the model with optimized settings.
- Evaluate the model and save the best version.
- Upload the trained model to Hugging Face.

A standalone Python script `main.py` is provided for running the model outside Google Colab. Ensure you have all dependencies installed by using the provided `requirements.txt` file.

## Model Architecture

The model is a custom CNN with the following structure:

- **Feature Extraction**: Multiple convolutional layers with ReLU activations and max pooling.
- **Classification Head**: Fully connected layers with dropout for regularization.
- **Training Optimizations**: Uses Adam optimizer, label smoothing, and ReduceLROnPlateau scheduler.

## Dataset

The dataset consists of images of planes and cars sourced from Hugging Face:

- **Planes Dataset**: [fearlixg/planes\_splitted](https://huggingface.co/datasets/fearlixg/planes_splitted)
- **Cars Dataset**: [fearlixg/cars\_splitted](https://huggingface.co/datasets/fearlixg/cars_splitted)

The dataset is balanced using data augmentation techniques such as cropping, flipping, rotation, color jitter, and random erasing.

## Training Process

- **Augmented Data Loading**: Uses PyTorch DataLoaders with transformations.
- **Mixup Augmentation**: Blends image-label pairs for improved generalization.
- **Early Stopping**: Stops training if validation accuracy does not improve.
- **Best Model Saving**: Stores the highest-performing model checkpoint.

## Performance Evaluation

The model is evaluated using:

- **Confusion Matrix**
- **Accuracy, Precision, Recall, and F1-Score**
- **ROC-AUC Score**

Sample results are visualized using Matplotlib and Seaborn.

## Uploading to Hugging Face

The script automates model uploading:

- Retrieves Hugging Face token from Google Colab.
- Creates or updates the repository.
- Saves model with a timestamped filename.
- Uploads using `huggingface_hub`.

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -m 'Add new feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a pull request.

## Contact

For questions or collaborations, reach out to me.

