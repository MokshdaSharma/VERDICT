import pytest
from pydantic import ValidationError
from verdict.graph.models import (
    Observation, FHIRProvenance, GuidelineProvenance, KnowledgeProvenance, DerivedProvenance, Edge
)
from verdict.graph.store import GraphStore

def test_node_creation_with_valid_provenance():
    # Should succeed
    obs = Observation(
        id="obs1",
        claim="Patient has high fever",
        provenance=FHIRProvenance(
            resource_type="Observation",
            resource_id="123",
            field="valueQuantity"
        ),
        w0=0.9,
        author="EvidenceAgent"
    )
    assert obs.id == "obs1"
    assert obs.type == "Observation"

def test_node_creation_fails_without_provenance():
    with pytest.raises(ValidationError):
        # Missing provenance field
        Observation(
            id="obs2",
            claim="Invalid node",
            w0=0.5,
            author="TestAgent"
        )

def test_node_creation_fails_with_invalid_provenance():
    with pytest.raises(ValidationError):
        # Invalid provenance format (string instead of Provenance object)
        Observation(
            id="obs3",
            claim="Invalid node",
            provenance="Just a string citation",  # type: ignore
            w0=0.5,
            author="TestAgent"
        )

def test_graph_store_add_and_retrieve():
    store = GraphStore()
    
    obs = Observation(
        id="obs1",
        claim="Fever",
        provenance=FHIRProvenance(
            resource_type="Observation",
            resource_id="123",
            field="valueQuantity"
        ),
        w0=1.0,
        author="EA"
    )
    store.add_node(obs)
    
    retrieved = store.get_node("obs1")
    assert retrieved is not None
    assert retrieved.claim == "Fever"
    
def test_graph_store_edges():
    store = GraphStore()
    
    obs1 = Observation(id="n1", claim="A", provenance=KnowledgeProvenance(document_id="doc1"), w0=0.8, author="A1")
    obs2 = Observation(id="n2", claim="B", provenance=KnowledgeProvenance(document_id="doc2"), w0=0.9, author="A2")
    
    store.add_node(obs1)
    store.add_node(obs2)
    
    store.add_edge(Edge(source_id="n1", target_id="n2", type="supports"))
    
    supporters = store.get_supporters("n2")
    assert supporters == ["n1"]
    
    attackers = store.get_attackers("n2")
    assert attackers == []
    
def test_graph_store_json_serialization():
    store = GraphStore()
    obs = Observation(id="n1", claim="A", provenance=GuidelineProvenance(rule_id="r1"), w0=0.5, author="A1")
    store.add_node(obs)
    
    json_str = store.to_json()
    assert "GuidelineRule" not in json_str # wait, type is Observation
    assert "Observation" in json_str
    
    new_store = GraphStore.from_json(json_str)
    assert new_store.get_node("n1").author == "A1"
