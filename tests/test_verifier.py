from verdict.verifier.backends import LLMEntailmentBackend
from verdict.verifier.gate import AdmissionGate
from verdict.graph.models import Observation, KnowledgeProvenance

def test_llm_backend_mock():
    # Since we have no client, it falls back to exact substring match
    backend = LLMEntailmentBackend(api_client=None)
    
    premise = "Patient has a temperature of 38.5 C."
    hypothesis = "temperature of 38.5 C"
    
    score = backend.verify(premise, hypothesis)
    assert score == 1.0
    
    hypothesis_false = "Patient has no fever"
    score_false = backend.verify(premise, hypothesis_false)
    assert score_false == 0.1

def test_admission_gate():
    backend = LLMEntailmentBackend(api_client=None)
    gate = AdmissionGate(verifier=backend, tau_entail=0.5)
    
    obs = Observation(
        id="obs1",
        claim="temperature of 38.5 C",
        provenance=KnowledgeProvenance(document_id="doc1"),
        w0=0.0,
        author="EA"
    )
    
    is_accepted, score = gate.check_node(obs, "Patient has a temperature of 38.5 C.")
    assert is_accepted is True
    assert score == 1.0
    assert obs.w0 == 1.0
    
    # Check rejection
    obs2 = Observation(
        id="obs2",
        claim="Patient has no fever",
        provenance=KnowledgeProvenance(document_id="doc1"),
        w0=0.0,
        author="EA"
    )
    is_accepted, score = gate.check_node(obs2, "Patient has a temperature of 38.5 C.")
    assert is_accepted is False
    assert score < 0.5
    assert len(gate.rejection_log) == 1
    assert gate.rejection_log[0]["node_id"] == "obs2"
