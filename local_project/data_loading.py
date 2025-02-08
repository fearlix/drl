from datasets import load_dataset

"""## ** Loading & Labeling**
This function loads two datasets from Hugging Face: one for planes and one for cars. It prints both datasets to check if they were loaded correctly and then returns them for further use.
"""

def load_datasets():
    """Loads the datasets from Hugging Face."""
    planes_dataset = load_dataset("fearlixg/planes_splitted")
    cars_dataset = load_dataset("fearlixg/cars_splitted")
    return planes_dataset, cars_dataset

"""This function adds labels to the datasets: **1 for planes** and **0 for cars**. It does this by applying a small helper function that adds a label field to each example. Then, it updates both the training and test sets of each dataset with the correct labels. Finally, it returns the updated datasets."""

def label_datasets(planes_dataset, cars_dataset):
    """Assigns labels: Planes (1), Cars (0)."""
    def add_label(example, label):
        example["label"] = label
        return example

    for dataset in [planes_dataset, cars_dataset]:
        label = 1 if dataset == planes_dataset else 0
        dataset["train"] = dataset["train"].map(lambda x: add_label(x, label))
        dataset["test"] = dataset["test"].map(lambda x: add_label(x, label))

    return planes_dataset, cars_dataset