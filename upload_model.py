# upload_model.py
from huggingface_hub import HfApi, login, upload_folder
import os
from dotenv import load_dotenv

# Load token from .env
load_dotenv()
login(token=os.getenv("HF_TOKEN"))

# Define repo and model folder
username = "akhil11y"
repo_name = "video-content-moderator"
model_dir = "models/best_model"

# Create repo (if not exists)
api = HfApi()
api.create_repo(repo_id=f"{username}/{repo_name}", private=False, exist_ok=True)

# Upload the entire model folder
upload_folder(
    repo_id=f"{username}/{repo_name}",
    folder_path=model_dir,
    path_in_repo="",  # Upload contents to the root of the repo
    repo_type="model",
    commit_message="Initial upload of trained video content moderation model"
)

print("✅ Model uploaded to https://huggingface.co/akhil11y/video-content-moderator")