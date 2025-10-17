# download_u2net.py
import os
import requests
from tqdm import tqdm

def download_file(url, filename):
    """Download file with progress bar"""
    if os.path.exists(filename):
        print(f"{filename} already exists")
        return
    
    print(f"Downloading {filename}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as file, tqdm(
        total=total_size, unit='B', unit_scale=True, desc=filename
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    model_url = "https://github.com/xuebinqin/U-2-Net/raw/master/saved_models/u2net/u2net.pth"
    download_file(model_url, "models/u2net.pth")