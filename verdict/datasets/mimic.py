import json
import os
from typing import List, Dict, Any
from verdict.datasets.interfaces import PatientRecord, DatasetAdapter

class LocalFhirPatient(PatientRecord):
    def __init__(self, patient_id: str, bundle: Dict[str, Any]):
        self.patient_id = patient_id
        self.bundle = bundle
        self.resources = [entry["resource"] for entry in bundle.get("entry", []) if "resource" in entry]
        
    def _get_resources(self, resource_type: str) -> List[Dict[str, Any]]:
        return [r for r in self.resources if r.get("resourceType") == resource_type]

    def get_patient_id(self) -> str:
        return self.patient_id

    def get_demographics(self) -> Dict[str, Any]:
        patients = self._get_resources("Patient")
        return patients[0] if patients else {}

    def get_observations(self) -> List[Dict[str, Any]]:
        return self._get_resources("Observation")

    def get_conditions(self) -> List[Dict[str, Any]]:
        return self._get_resources("Condition")

    def get_medications(self) -> List[Dict[str, Any]]:
        return self._get_resources("MedicationRequest")


class MimicFhirLoader(DatasetAdapter):
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.patient_files: Dict[str, str] = {}
        self._scan_data()
        
    def _scan_data(self):
        manifest_path = os.path.join(self.data_dir, "manifest.json")
        source_dir = os.path.join("data", "mimic", "mimic-iv-clinical-database-demo-on-fhir-2.1.0", "fhir")
        
        if not os.path.exists(manifest_path):
            print("MIMIC-IV converted bundles not found or incomplete. Running conversion (this will be cached)...")
            # Fallback to absolute/relative import
            import sys
            if os.path.abspath("scripts") not in sys.path:
                sys.path.insert(0, os.path.abspath("scripts"))
            from convert_mimic import convert
            convert(source_dir, self.data_dir)
            print("Using newly converted MIMIC-IV data.")
        else:
            print("Using cached MIMIC-IV conversion.")
            
        if not os.path.exists(self.data_dir):
            return
            
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".json"):
                # For Mimic and Synthea, filenames might be complex. Just use filename without .json as ID.
                pat_id = filename.replace(".json", "")
                self.patient_files[pat_id] = os.path.join(self.data_dir, filename)

    def get_patient(self, patient_id: str) -> PatientRecord:
        if patient_id not in self.patient_files:
            raise KeyError(f"Patient {patient_id} not found.")
        path = self.patient_files[patient_id]
        with open(path, "r", encoding="utf-8") as f:
            bundle = json.load(f)
            return LocalFhirPatient(patient_id, bundle)
        
    def get_all_patient_ids(self) -> List[str]:
        return list(self.patient_files.keys())
