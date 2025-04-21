# content_moderator.py
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from huggingface_hub import login
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from dotenv import load_dotenv
import time

# Load Hugging Face token from .env
load_dotenv()
login(token=os.getenv("HF_TOKEN"))

class VideoFrameDataset(Dataset):
    def __init__(self, frames, labels, feature_extractor):
        self.frames = frames
        self.labels = labels
        self.feature_extractor = feature_extractor
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        frame = self.frames[idx]
        label = self.labels[idx]
        image = Image.fromarray(frame)
        inputs = self.feature_extractor(image, return_tensors="pt")
        return {
            'pixel_values': inputs['pixel_values'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class ContentModerator:
    def __init__(self, model_name="akhil11y/video-content-moderator", train_mode=False):
        start_time = time.time()
        print(f"Initializing ContentModerator from {model_name}...")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device} ({time.time() - start_time:.2f}s elapsed)")

        print(f"Downloading/loading feature extractor from {model_name}...")
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        print(f"Feature extractor loaded ({time.time() - start_time:.2f}s elapsed)")

        if train_mode:
            print(f"Downloading/loading model (train mode) from {model_name}...")
            self.model = AutoModelForImageClassification.from_pretrained(
                model_name,
                num_labels=2,
                ignore_mismatched_sizes=True
            ).to(self.device)
        else:
            print(f"Downloading/loading model (inference mode) from {model_name}...")
            self.model = AutoModelForImageClassification.from_pretrained(
                model_name,
                num_labels=2
            ).to(self.device)
            self.model.eval()
        print(f"Model loaded from {model_name} ({time.time() - start_time:.2f}s elapsed)")

    def analyze_frames(self, frames):
        results = []
        dataset = VideoFrameDataset(frames, [0] * len(frames), self.feature_extractor)
        dataloader = DataLoader(dataset, batch_size=32)

        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                pixel_values = batch['pixel_values'].to(self.device)
                outputs = self.model(pixel_values)
                predictions = torch.softmax(outputs.logits, dim=1)

                for pred in predictions:
                    violence_prob = pred[1].item()
                    flagged = violence_prob > 0.3
                    results.append({
                        'flagged': flagged,
                        'reason': "Detected violence" if flagged else "No inappropriate content detected",
                        'confidence': violence_prob if flagged else 1 - violence_prob
                    })

        return results