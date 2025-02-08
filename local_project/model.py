import torch
import torch.nn as nn


"""## **Define model functions**

The setup_device function checks if a GPU is available and returns the appropriate device, either CUDA or CPU.

The ConvNN class defines a convolutional neural network designed for binary classification between planes and cars. It consists of multiple convolutional layers with ReLU activations, max pooling, and batch normalization to extract features from images. The network includes an adaptive average pooling layer followed by fully connected layers with dropout and LeakyReLU activations to improve generalization.

The setup_model function initializes this network and moves it to the selected device. The setup_training_components function sets up the loss function with label smoothing, the Adam optimizer with weight decay, and a learning rate scheduler that reduces the learning rate when performance plateaus. The mixup_criterion function implements a loss calculation method for mixup augmentation by interpolating labels for better model generalization.

"""

class ConvNN(nn.Module):
    """Custom Convolutional Neural Network for binary classification (plane vs. car)."""

    def __init__(self, num_classes: int = 2):
        super(ConvNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.55),
            nn.Linear(256 * 6 * 6, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.55),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 2)
        )


    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def setup_device():
    """Returns the available device (GPU or CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")



def setup_model(device):
    """Initializes and moves the ConvNN model to the specified device."""
    model = ConvNN(num_classes=2)
    return model.to(device)
