"""
Script to download REFIT dataset.
Note: REFIT requires registration, so this provides instructions
and alternative access via public mirror if available.
"""

import os
import requests
from tqdm import tqdm
import zipfile

def download_refit_instructions():
    """Print instructions for accessing REFIT dataset."""
    instructions = """
    ================================================================
    REFIT ELECTRICAL LOAD MEASUREMENT DATASET - ACCESS INSTRUCTIONS
    ================================================================
    
    The REFIT dataset requires registration due to ethical agreements.
    
    OFFICIAL SOURCE:
    1. Go to: https://repository.lboro.ac.uk/articles/dataset/REFIT_Electrical_Load_Measurement/2070091
    2. Click "Download" (requires free registration)
    3. Place downloaded files in ./data/raw/
    
    ALTERNATIVE (if you have academic credentials):
    Some institutions provide mirrors. Check your university library.
    
    After downloading, run:
        python data/preprocess.py
    
    Expected file structure:
        data/raw/
            ├── Household_1.csv
            ├── Household_2.csv
            └── ...
    
    ================================================================
    """
    print(instructions)
    
    # Create directories
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    # Create placeholder file
    with open("data/raw/README.txt", "w") as f:
        f.write("Place REFIT CSV files here after downloading from official source.\n")
        f.write("See download_data.py instructions for details.\n")

if __name__ == "__main__":
    download_refit_instructions()
    print("\nDirectories created. Follow instructions above to complete data setup.")
