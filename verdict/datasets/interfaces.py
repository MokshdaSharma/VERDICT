from abc import ABC, abstractmethod
from typing import List, Dict, Any

class PatientRecord(ABC):
    @abstractmethod
    def get_patient_id(self) -> str:
        pass
        
    @abstractmethod
    def get_demographics(self) -> Dict[str, Any]:
        pass
        
    @abstractmethod
    def get_observations(self) -> List[Dict[str, Any]]:
        pass
        
    @abstractmethod
    def get_conditions(self) -> List[Dict[str, Any]]:
        pass
        
    @abstractmethod
    def get_medications(self) -> List[Dict[str, Any]]:
        pass

class DatasetAdapter(ABC):
    @abstractmethod
    def get_patient(self, patient_id: str) -> PatientRecord:
        pass
        
    @abstractmethod
    def get_all_patient_ids(self) -> List[str]:
        pass
