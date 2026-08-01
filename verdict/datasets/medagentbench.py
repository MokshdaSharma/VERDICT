import requests
from typing import List, Dict, Any
from verdict.datasets.interfaces import PatientRecord, DatasetAdapter

class MedAgentBenchPatient(PatientRecord):
    def __init__(self, patient_id: str, base_url: str):
        self.patient_id = patient_id
        self.base_url = base_url
        
    def _fetch(self, resource_type: str) -> List[Dict[str, Any]]:
        # A real implementation would query the FHIR server
        # e.g., GET {base_url}/{resource_type}?subject={patient_id}
        url = f"{self.base_url}/{resource_type}?subject=Patient/{self.patient_id}"
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()
            if data.get("resourceType") == "Bundle":
                return [entry["resource"] for entry in data.get("entry", [])]
            return [data]
        except Exception as e:
            print(f"Error fetching {resource_type} for {self.patient_id}: {e}")
            return []

    def get_patient_id(self) -> str:
        return self.patient_id

    def get_demographics(self) -> Dict[str, Any]:
        try:
            response = requests.get(f"{self.base_url}/Patient/{self.patient_id}", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception:
            return {}

    def get_observations(self) -> List[Dict[str, Any]]:
        return self._fetch("Observation")

    def get_conditions(self) -> List[Dict[str, Any]]:
        return self._fetch("Condition")

    def get_medications(self) -> List[Dict[str, Any]]:
        return self._fetch("MedicationRequest")


class MedAgentBenchClient(DatasetAdapter):
    def __init__(self, fhir_server_url: str = "http://localhost:8080/fhir"):
        self.fhir_server_url = fhir_server_url
        
    def get_patient(self, patient_id: str) -> PatientRecord:
        return MedAgentBenchPatient(patient_id, self.fhir_server_url)
        
    def get_all_patient_ids(self) -> List[str]:
        # Typically requires querying the Patient endpoint without filters, or parsing a known list.
        # For the POC, this would return a hardcoded list or fetch a subset.
        try:
            response = requests.get(f"{self.fhir_server_url}/Patient", timeout=5)
            response.raise_for_status()
            data = response.json()
            return [entry["resource"]["id"] for entry in data.get("entry", [])]
        except Exception:
            return []
