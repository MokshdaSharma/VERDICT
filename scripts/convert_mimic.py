import os
import gzip
import json
from collections import defaultdict

def extract_patient_id(resource):
    rt = resource.get("resourceType")
    if rt == "Patient":
        return resource.get("id")
    
    # Try subject
    subject = resource.get("subject", {})
    if isinstance(subject, dict):
        ref = subject.get("reference", "")
        if ref.startswith("Patient/"):
            return ref.replace("Patient/", "")
            
    # Try patient
    patient = resource.get("patient", {})
    if isinstance(patient, dict):
        ref = patient.get("reference", "")
        if ref.startswith("Patient/"):
            return ref.replace("Patient/", "")
            
    return None

def convert(data_dir: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    
    patient_resources = defaultdict(list)
    
    print(f"Scanning {data_dir} for .ndjson.gz files...")
    for filename in os.listdir(data_dir):
        if not filename.endswith(".ndjson.gz"):
            continue
            
        print(f"Processing {filename}...")
        path = os.path.join(data_dir, filename)
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    resource = json.loads(line)
                    pid = extract_patient_id(resource)
                    if pid:
                        patient_resources[pid].append(resource)
                except Exception as e:
                    pass
                    
    print(f"Found {len(patient_resources)} patients. Writing bundles to {out_dir}...")
    
    for pid, resources in patient_resources.items():
        bundle = {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [{"resource": r} for r in resources]
        }
        out_path = os.path.join(out_dir, f"{pid}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(bundle, f)

    print("Conversion complete!")
    
    # Write manifest
    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"status": "completed", "patient_count": len(patient_resources)}, f)

if __name__ == "__main__":
    mimic_dir = os.path.join("data", "mimic", "mimic-iv-clinical-database-demo-on-fhir-2.1.0", "fhir")
    out_dir = os.path.join("data", "mimic", "bundles")
    convert(mimic_dir, out_dir)
