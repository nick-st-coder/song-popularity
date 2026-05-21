import zipfile
import os

zip_path = "Data/spotify-music-dataset.zip"
extract_to = "data/"

with zipfile.ZipFile(zip_path, "r") as z:
    z.extractall(extract_to)

os.remove(zip_path)
print("Extracted and deleted zip file.")