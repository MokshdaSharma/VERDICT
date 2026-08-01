from typing import List, Literal, Union
from pydantic import BaseModel, Field

class FHIRProvenance(BaseModel):
    type: Literal["fhir"] = "fhir"
    resource_type: str
    resource_id: str
    field: str

class GuidelineProvenance(BaseModel):
    type: Literal["guideline"] = "guideline"
    rule_id: str

class KnowledgeProvenance(BaseModel):
    type: Literal["knowledge"] = "knowledge"
    document_id: str

class DerivedProvenance(BaseModel):
    type: Literal["derived"] = "derived"
    source_node_ids: List[str] = Field(min_length=1)

Provenance = Union[FHIRProvenance, GuidelineProvenance, KnowledgeProvenance, DerivedProvenance]

class BaseNode(BaseModel):
    id: str
    claim: str
    # Enforcing strict provenance at the data model level.
    # It is impossible to instantiate a valid node without it.
    provenance: Provenance
    w0: float = Field(ge=0.0, le=1.0)
    author: str
    type: str

class Observation(BaseNode):
    type: Literal["Observation"] = "Observation"

class Concept(BaseNode):
    type: Literal["Concept"] = "Concept"

class GuidelineRule(BaseNode):
    type: Literal["GuidelineRule"] = "GuidelineRule"

class Hypothesis(BaseNode):
    type: Literal["Hypothesis"] = "Hypothesis"

class PersonalizationConstraint(BaseNode):
    type: Literal["PersonalizationConstraint"] = "PersonalizationConstraint"

class Recommendation(BaseNode):
    type: Literal["Recommendation"] = "Recommendation"

class CounterEvidence(BaseNode):
    type: Literal["CounterEvidence"] = "CounterEvidence"

AnyNode = Union[Observation, Concept, GuidelineRule, Hypothesis, PersonalizationConstraint, Recommendation, CounterEvidence]

class Edge(BaseModel):
    source_id: str
    target_id: str
    type: Literal["supports", "attacks"]
