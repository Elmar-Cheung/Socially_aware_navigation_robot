import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path("C:/Users/Elmar/Desktop/MTP/All_models/Gen_PS_map/.env"))

key = os.environ.get("WANDB_API_KEY")
print(f"Key found: {bool(key)}")
print(f"Key starts with: {key[:10] if key else 'NONE'}")

import wandb
print(f"wandb version: {wandb.__version__}")

# Test if api.wandb.ai is reachable
import urllib.request
try:
    urllib.request.urlopen("https://api.wandb.ai", timeout=5)
    print("Network: wandb.ai is reachable")
except Exception as e:
    print(f"Network ERROR: {e}")

# Test offline mode (bypasses auth entirely)
wandb.init(
    project="social_nav",
    settings=wandb.Settings(
        api_key=os.environ["WANDB_API_KEY"],
        mode="online"
    ),
    config={
        "algorithm": "SAC",
        "total_timesteps": 10000,
        "num_humans": 2,
        "width": 10,
        "height": 10,
    }
)
print("offline init worked")
wandb.finish()