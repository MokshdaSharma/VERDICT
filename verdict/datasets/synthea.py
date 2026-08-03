from verdict.datasets.mimic import LocalFhirPatient, MimicFhirLoader
import os

class SyntheaLoader(MimicFhirLoader):
    """
    Synthea generates FHIR bundles per patient.
    We can reuse the Mimic loader logic since both just read local JSON bundles.
    """
    def _scan_data(self):
        if not os.path.exists(self.data_dir):
            return
            
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".json"):
                pat_id = filename.replace(".json", "")
                self.patient_files[pat_id] = os.path.join(self.data_dir, filename)
