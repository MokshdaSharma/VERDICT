from verdict.datasets.mimic import LocalFhirPatient, MimicFhirLoader

class SyntheaLoader(MimicFhirLoader):
    """
    Synthea generates FHIR bundles per patient.
    We can reuse the Mimic loader logic since both just read local JSON bundles.
    """
    pass
