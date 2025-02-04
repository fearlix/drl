Cars vs Planes Classification Using ResNet50

This project trains a ResNet50 model to classify images of cars and planes using datasets from Hugging Face.

Features

Uses Hugging Face Datasets for easy data access.

Fine-tunes ResNet50 while freezing early layers for transfer learning.

Applies MixUp augmentation for better generalization.

Uploads trained models to Hugging Face for easy sharing.

Installation

Clone the Repository

git clone https://github.com/yourusername/cars-vs-planes.git
cd cars-vs-planes

Install Dependencies

Using pip

pip install -r requirements.txt

Using Conda

conda env create -f environment.yml
conda activate cars_vs_planes

Setup Hugging Face Authentication

Log in to Hugging Face to upload/download models:

huggingface-cli login

Running the Model

To train the model, run:

python main.py

Expected Output

After training, you should see output like this:

Training the model...
Epoch 1/4: 100%|██████████| 500/500 [07:08<00:00,  1.17it/s]
Train Accuracy: 66.64%

Validating the trained model...
Test Accuracy: 97.35%
Precision: 0.99
Recall: 0.97
F1-Score: 0.98
ROC-AUC Score: 1.00

Uploading the trained model...
Model uploaded to Hugging Face!

Project Structure

cars-vs-planes/
│── main.py               # Main script to train, validate, and upload model
│── model.py              # Defines ResNet50 model architecture
│── dataset.py            # Handles dataset loading & transformations
│── train.py              # Contains training functions
│── validate.py           # Handles model evaluation
│── upload.py             # Handles Hugging Face model uploads
│── requirements.txt      # List of dependencies
│── environment.yml       # Conda environment file
│── README.md             # Project documentation

Sample Input & Expected Output

To ensure replicability, the project includes:

Sample images in the samples/ directory.

Expected results for verification.

Sample Folder Structure

samples/
│── car_1.jpg
│── plane_1.jpg
│── expected_output.txt

Example expected_output.txt

Predicted Class for car_1.jpg: Car
Predicted Class for plane_1.jpg: Plane

Contributing

Feel free to open issues or pull requests to improve this project.

License

This project is licensed under the MIT License.

Contact

For any questions, reach out at your.email@example.com.

