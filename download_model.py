#!/usr/bin/env python3
"""
Download the DSRU model checkpoint from Google Drive.
"""
import subprocess
import sys
from pathlib import Path

def download_with_gdown():
    """Download using gdown library."""
    try:
        import gdown
    except ImportError:
        print("Installing gdown...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        import gdown
    
    # Your Google Drive URL
    url = "https://drive.google.com/file/d/1oZarHzA7PwSij6aBGOQaEHCnB100St3H/view?usp=sharing"
    output = "model.pt"
    
    print(f"Downloading DSRU model checkpoint...")
    print(f"This may take a while depending on your connection speed.")
    
    try:
        gdown.download(url, output, quiet=False, fuzzy=True)
        
        # Verify download
        if Path(output).exists():
            file_size = Path(output).stat().st_size / (1024 * 1024 * 1024)  # GB
            print(f"\nDownload complete!")
            print(f"File size: {file_size:.2f} GB")
        else:
            print("Error: Download failed - file not found")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error downloading: {e}")
        print("\nTry downloading manually from:")
        print(url)
        sys.exit(1)

def main():
    # Check if file already exists
    if Path("model.pt").exists():
        response = input("model.pt already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Download cancelled.")
            return
    
    download_with_gdown()

if __name__ == "__main__":
    main()
