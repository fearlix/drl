import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
import time

def plot_confusion_matrix(true_labels, pred_labels):
    """Plots confusion matrix after training."""
    cm = confusion_matrix(true_labels, pred_labels)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()



def setup_training_components(model, train_loader):
    """Initializes loss function, optimizer, and scheduler for stability and efficiency."""

    criterion = nn.CrossEntropyLoss(label_smoothing=0.2)

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.00001)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)

    return criterion, optimizer, scheduler

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def plot_accuracy(history):
    """Plots training & testing accuracy over epochs."""
    epochs = range(1, len(history["train_acc"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_acc"], label="Train Accuracy", marker="o", linestyle="-")
    plt.plot(epochs, history["test_acc"], label="Test Accuracy", marker="s", linestyle="--")

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training & Testing Accuracy Over Epochs")
    plt.legend()
    plt.grid()
    plt.show()


"""## **7.  Define Training**

This function trains a deep learning model while tracking accuracy and plotting results. It loops through a set number of epochs, training the model using the provided optimizer and loss function. During each epoch, it calculates training accuracy by comparing predicted labels with actual labels. After training, it evaluates the model on the test set without updating weights. It stores test accuracy, predictions, and labels for later analysis. If the model improves, it saves the best version. If performance does not improve for a set number of epochs, it stops training early. After training, it plots accuracy trends and generates a confusion matrix for performance visualization.
"""

def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, device, epochs=10, patience=5):
    """
    Trains a model while tracking accuracy, showing progress percentage, estimating time left, and plotting results.

    Returns:
    - model: Best trained model.
    - history: Dictionary containing accuracy values for plotting.
    """

    best_accuracy = 0
    patience_counter = 0
    history = {"train_acc": [], "test_acc": []}

    print(f"\n Training Model for {epochs} Epochs...\n")

    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        correct_train = 0
        total_train = 0
        num_batches = len(train_loader)

        print(f"\nEpoch {epoch+1}/{epochs} Training...")

        # Training loop with progress percentage
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

            # Live progress update
            progress = int((batch_idx + 1) / num_batches * 100)
            time_per_batch = (time.time() - epoch_start) / (batch_idx + 1)
            estimated_time = (num_batches - (batch_idx + 1)) * time_per_batch

            print(f"\rTraining Progress: {progress}% | Estimated Time Left: {estimated_time:.2f}s", end="")

        train_accuracy = correct_train / total_train
        history["train_acc"].append(train_accuracy)

        model.eval()
        correct_test = 0
        total_test = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                correct_test += (predicted == labels).sum().item()
                total_test += labels.size(0)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        test_accuracy = correct_test / total_test
        history["test_acc"].append(test_accuracy)

        epoch_duration = time.time() - epoch_start
        print(f"\nEpoch {epoch+1}/{epochs} Completed: Train Acc: {train_accuracy:.4f}, Test Acc: {test_accuracy:.4f} | Time Taken: {epoch_duration:.2f}s")

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), "best_checkpoint.model")
            patience_counter = 0  #
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}. Best test accuracy: {best_accuracy:.4f}")
            break

        if scheduler:
            scheduler.step(test_accuracy)

    total_training_time = time.time() - start_time
    print(f"\n Training Completed in {total_training_time:.2f}s")

    plot_accuracy(history)
    plot_confusion_matrix(all_labels, all_preds)

    return model
