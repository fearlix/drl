import torch
from sklearn.metrics import (
    confusion_matrix, classification_report, precision_score,
    recall_score, f1_score, roc_auc_score
)
import torch.nn.functional as F

"""## Define Validation**

This function evaluates a trained model using a test dataset and prints performance metrics. It switches the model to evaluation mode and processes the test images without updating weights. It computes predictions by selecting the class with the highest probability and extracts confidence scores for class 1. The function calculates overall accuracy by comparing predictions with actual labels. It generates a confusion matrix and automatically detects the number of unique classes to create class labels dynamically. It computes precision, recall, F1-score, and ROC-AUC score to assess classification performance. Finally, it prints a detailed classification report and visualizes the confusion matrix using a heatmap.
"""

def validate_model(model, test_loader, device):
    """Evaluates the model and prints performance metrics."""
    model.eval()
    correct, total = 0, 0
    all_labels, all_predictions, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)
            probs = F.softmax(outputs, dim=1)[:, 1]

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    accuracy = 100 * correct / total
    conf_matrix = confusion_matrix(all_labels, all_predictions)

    # Automatically detect number of classes
    unique_classes = sorted(set(all_labels))
    num_classes = len(unique_classes)

    # Dynamically create class names
    target_names = [f"Class {i}" for i in unique_classes]

    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
    roc_auc = roc_auc_score(all_labels, all_probs, multi_class="ovr")

    print(f"\nTest Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")
    print(f"ROC-AUC Score: {roc_auc:.2f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, target_names=target_names))

    return accuracy