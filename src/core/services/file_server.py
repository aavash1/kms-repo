# src/core/services/file_server.py

import requests
import os

def fetch_file_from_server(file_path, save_dir="/tmp"):
    """
    Connect to the file server and retrieve the file based on the given file path.
    """
    file_url = f"http://file-server.local/files/{file_path}"  # Replace with actual file server URL
    response = requests.get(file_url)
    
    if response.status_code == 200:
        save_path = os.path.join(save_dir, os.path.basename(file_path))
        with open(save_path, "wb") as file:
            file.write(response.content)
        return save_path
    else:
        raise Exception(f"Failed to fetch file from server. Status Code: {response.status_code}")
