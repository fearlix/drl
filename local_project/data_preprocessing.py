from datasets import concatenate_datasets, DatasetDict
from torchvision import transforms
from PIL import Image

def merge_and_balance_datasets(planes_dataset, cars_dataset):
    """Merges and balances datasets by augmenting and oversampling the minority class."""
    train_planes, test_planes = planes_dataset["train"], planes_dataset["test"]
    train_cars, test_cars = cars_dataset["train"], cars_dataset["test"]

    num_planes, num_cars = len(train_planes), len(train_cars)

    def augment_example(example):
        augmentation = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ])

        image = example["image"]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        example["image"] = augmentation(image)
        return example

    if num_planes > num_cars:
        additional_cars = train_cars.shuffle(seed=42).map(augment_example).select(range(num_planes - num_cars))
        balanced_train_dataset = concatenate_datasets([train_planes, train_cars, additional_cars])
    else:
        additional_planes = train_planes.shuffle(seed=42).map(augment_example).select(range(num_cars - num_planes))
        balanced_train_dataset = concatenate_datasets([train_planes, train_cars, additional_planes])

    balanced_test_dataset = concatenate_datasets([test_planes, test_cars])

    return DatasetDict({"train": balanced_train_dataset, "test": balanced_test_dataset})

def get_transforms():
    """Returns train and test transformations with optimized simplicity."""
    transform_train = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.ToTensor(),
        transforms.RandomRotation(degrees=30),
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.3), ratio=(0.2, 3.0)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return transform_train, transform_test
