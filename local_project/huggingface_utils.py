from huggingface_hub import (
    login, whoami, create_repo, upload_file,
    list_repo_files
)
import time
import torch
import json

"""## ** Uploading best model**"""
def get_huggingface_token():
    """Fetches the Hugging Face token from a JSON file, logs in, and returns the token."""
    json_path = "secrets.json"

    try:
        with open(json_path, "r") as file:
            secrets = json.load(file)
            hf_token = secrets.get("HF_TOKEN")
    except FileNotFoundError:
        print("secrets.json file not found.")
        return None

    if hf_token:
        login(token=hf_token)
        print("Successfully logged in to Hugging Face!")
        return hf_token
    else:
        print("Hugging Face token not found in secrets.json.")
        return None

def get_repo_id_and_model(hf_token, model_name):
    """Generates the repository ID and model filename for Hugging Face models."""
    try:
        user_info = whoami(token=hf_token)
        username = user_info.get("name") or user_info.get("login", "Unknown User")

        if username == "Unknown User":
            print("Error: Could not retrieve username. Please check your token.")
            return None, None

        repo_id = f"{username}/{model_name}"
        model_filename = f"best_{model_name}"

        print("Repo ID:", repo_id)
        print("Model Filename:", model_filename)

        return repo_id, model_filename
    except Exception as e:
        print(f"Error fetching repo ID: {e}")
        return None, None


def upload_new_model_with_timestamp(model, repo_id, model_name, hf_token=None):
    """Uploads the model with a timestamp to Hugging Face."""
    if hf_token:
        login(token=hf_token)
    else:
        print("Hugging Face token is missing.")
        return

    try:
        list_repo_files(repo_id, token=hf_token)
    except Exception:
        create_repo(repo_id, exist_ok=True, token=hf_token)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_filename = f"{model_name}_{timestamp}.pth"

    torch.save(model.state_dict(), model_filename)
    upload_file(path_or_fileobj=model_filename, path_in_repo=model_filename, repo_id=repo_id, token=hf_token)
    print(f"Model uploaded as {repo_id}/{model_filename}")
