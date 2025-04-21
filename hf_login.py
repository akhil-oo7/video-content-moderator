# hf_login.py
from huggingface_hub import login
import os
from dotenv import load_dotenv

load_dotenv()
login(token=os.getenv("HF_TOKEN"))
print("✅ Logged into Hugging Face!")