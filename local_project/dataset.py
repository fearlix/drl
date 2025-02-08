from torch.utils.data import Dataset, DataLoader
from PIL import Image

"""## Create DataLoaders**

This class creates a custom dataset for Hugging Face images so they can be used in PyTorch. Additionally the function creates DataLoaders for training and testing by applying the specified transformations that were created beforehand to the dataset.
"""

class HuggingFaceImageDataset(Dataset):
    """Custom dataset class for Hugging Face datasets."""
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]

        if not hasattr(image, "convert"):
            image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        return image, item["label"]

def prepare_datasets(final_dataset, transform_train, transform_test):
    """Creates DataLoaders for training and testing."""
    train_data = HuggingFaceImageDataset(final_dataset["train"], transform=transform_train)
    test_data = HuggingFaceImageDataset(final_dataset["test"], transform=transform_test)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    return train_loader, test_loader
