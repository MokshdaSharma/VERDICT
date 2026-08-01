import requests
import zipfile
import os
import io

url = "https://physionet.org/static/published-projects/mimic-iv-fhir-demo/mimic-iv-fhir-demo-2.1.0.zip"

print(f"Downloading {url}...")
response = requests.get(url, stream=True)
response.raise_for_status()

# Create data/mimic directory if it doesn't exist
os.makedirs(os.path.join("data", "mimic"), exist_ok=True)

# Unzip directly from the stream
with zipfile.ZipFile(io.BytesIO(response.content)) as z:
    # the zip file might have a root folder mimic-iv-clinical-database-demo-on-fhir-2.1.0
    # extract everything into data/mimic
    z.extractall(os.path.join("data", "mimic"))

print("Download and extraction to data/mimic complete!")
