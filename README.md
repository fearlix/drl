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
2. Open [Colab Notebook](https://colab.research.google.com/drive/1Rb2ScqetPYqYMxRaXiUJ2TL1W5V4BDdz).
3. Run the notebook cell by cell.
4. Authenticate with Hugging Face when prompted.

To execute the pipeline in Google Colab:

1. Download the DSTI.
2. Open [Colab Notebook](https://colab.research.google.com/drive/1Rb2ScqetPYqYMxRaXiUJ2TL1W5V4BDdz)
3. Run the notebook cell by cell.
4. Authenticate with Hugging Face when prompted.

### Running Locally with Python Script

A standalone Python script `local_project/DSTI.py` is provided for running the model outside Google Colab. Ensure you have all dependencies installed by using the provided `local_project/requirements.txt` file.

#### 1. Clone the Repository

```bash
git clone https://github.com/your-repo/deep-learning-image-classification.git
cd deep-learning-image-classification
```

#### 2. Install Dependencies

```bash
pip install -r local_project/requirements.txt
```

#### 3. Run the Model

```bash
python local_project/DSTI.py
```

This will:
- Authenticate Hugging Face.
- Load and preprocess datasets.
- Train the model with optimized settings.
- Evaluate the model and save the best version.
- Upload the trained model to Hugging Face.

A standalone Python script `DSTI.py` is provided for running the model outside Google Colab. Ensure you have all dependencies installed by using the provided `requirements.txt` file.

#### 1. Clone the Repository

```bash
git clone https://github.com/your-repo/deep-learning-image-classification.git
cd deep-learning-image-classification
```

#### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 3. Run the Model

```bash
python DSTI.py
```

This will:

- Authenticate Hugging Face.
- Load and preprocess datasets.
- Train the model with optimized settings.
- Evaluate the model and save the best version.
- Upload the trained model to Hugging Face.

### 1. Clone the Repository

```bash
git clone https://github.com/your-repo/deep-learning-image-classification.git
cd deep-learning-image-classification
```

### 2. Run the Training Pipeline

The training process is managed through the `main()` function:

```python
python DSTI.ipynb
```

This will:

- Authenticate Hugging Face.
- Load and preprocess datasets.
- Train the model with optimized settings.
- Evaluate the model and save the best version.
- Upload the trained model to Hugging Face.

### 3. Validate the Model

```python
python validate_model.py
```

This will run evaluation metrics and print the classification report.

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

## Running on Google Colab

To execute the pipeline in Google Colab:

1. Open [Colab Notebook](https://colab.research.google.com/drive/1Rb2ScqetPYqYMxRaXiUJ2TL1W5V4BDdz)
2. Run the notebook cell by cell.
3. Authenticate with Hugging Face when prompted.

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -m 'Add new feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or collaborations, reach out to [[your-email@example.com](mailto\:your-email@example.com)].

