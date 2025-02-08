from drl.local_project import (
    load_datasets, label_datasets, merge_and_balance_datasets,
    get_transforms, prepare_datasets, train_model, setup_model, setup_training_components,
    validate_model, get_huggingface_token, get_repo_id_and_model, upload_new_model_with_timestamp, setup_device
)

"""## ** Main Function**

This function runs the full training and evaluation process step by step.
"""

def main():

    # Variables
    EPOCHS = 1
    MODEL_NAME = "cars_vs_planes_model"

    """Runs the full training and evaluation pipeline."""
    print("### Getting Hugging Face authentication ###")
    hf_token = get_huggingface_token()

    repo_id, model_filename = get_repo_id_and_model(hf_token, MODEL_NAME)

    print(repo_id)
    device = setup_device()

    print("### Loading datasets ###")
    planes_dataset, cars_dataset = load_datasets()

    print("### Labeling datasets ###")
    planes_dataset, cars_dataset = label_datasets(planes_dataset, cars_dataset)

    print("### Merging and balancing datasets ###")
    final_dataset = merge_and_balance_datasets(planes_dataset, cars_dataset)

    print("### Applying data transformations ###")
    transform_train, transform_test = get_transforms()

    print("### Creating DataLoaders ###")
    train_loader, test_loader = prepare_datasets(final_dataset, transform_train, transform_test)

    print("### Initializing model ###")
    model = setup_model(device)
    criterion, optimizer, scheduler = setup_training_components(model, train_loader)

    print("### Training the model ###")
    trained_model = train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, device, epochs=EPOCHS)

    print("### validating the best model ###")
    validate_model(trained_model, test_loader, device)

    print("### Uploading the trained model ###")
    upload_new_model_with_timestamp(trained_model, repo_id, model_filename, hf_token)

if __name__ == "__main__":
    main()