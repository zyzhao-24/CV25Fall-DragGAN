import os
import sys
import json
import requests
from tqdm import tqdm

def download_file(url: str, filename: str, download_dir: str):
    """Download a file if it does not already exist."""

    try:
        filepath = os.path.join(download_dir, filename)
        content_length = int(requests.head(url).headers.get("content-length", 0))

        # If file already exists and size matches, skip download
        if os.path.isfile(filepath) and os.path.getsize(filepath) == content_length:
            print(f"{filepath} already exists. Skipping download.")
            return
        if os.path.isfile(filepath) and os.path.getsize(filepath) != content_length:
            print(f"{filepath} already exists but size does not match. Redownloading.")
        else:
            print(f"Downloading {filename} from {url}")

        # Start download, stream=True allows for progress tracking
        response = requests.get(url, stream=True)

        # Check if request was successful
        response.raise_for_status()

        # Create progress bar
        total_size = int(response.headers.get('content-length', 0))
        progress_bar = tqdm(
            total=total_size, 
            unit='iB', 
            unit_scale=True, 
            ncols=70, 
            file=sys.stdout
        )

        # Write response content to file
        with open(filepath, 'wb') as f:
            for data in response.iter_content(chunk_size=1024):
                f.write(data)
                progress_bar.update(len(data))  # Update progress bar

        # Close progress bar
        progress_bar.close()

        # Error handling for incomplete downloads
        if total_size != 0 and progress_bar.n != total_size:
            print("ERROR, something went wrong while downloading")
            raise Exception()


    except Exception as e:
        print(f"An error occurred: {e}")

def unzip_file(zip_path: str, extract_to: str):
    """Unzip a zip file to the specified directory."""
    import zipfile

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Extracted {zip_path} to {extract_to}")
    except Exception as e:
        print(f"An error occurred while unzipping: {e}")

def main():
    """Main function to download files from URLs in a config file."""
    
    # Get JSON config file path
    script_dir = os.path.dirname(os.path.realpath(__file__))
    config_file_path = os.path.join(script_dir, "download_models.json")

    # Set download directory
    download_dir = os.path.join(os.path.dirname(script_dir), "checkpoints")
    os.makedirs(download_dir, exist_ok=True)

    # Load URL and filenames from JSON
    with open(config_file_path, "r") as f:
        config = json.load(f)

    # Download each file specified in config
    for url, filename in config.items():
        if os.path.exists(os.path.join(download_dir, filename)):
            print(f"{filename} already exists in {download_dir}. Skipping download.")
        else:
            download_file(url, filename, download_dir)

        # If the downloaded file is a zip, unzip it
        if filename.endswith(".zip"):
            unzip_file(os.path.join(download_dir, filename), download_dir)
            files = os.listdir(os.path.join(download_dir, filename[:-4]))
            for f in files:
                if os.path.exists(os.path.join(download_dir, f)):
                    os.remove(os.path.join(download_dir, f))
                    os.rename(os.path.join(download_dir, filename[:-4], f), os.path.join(download_dir, f))
            os.rmdir(os.path.join(download_dir, filename[:-4]))

if __name__ == "__main__":
    main()
