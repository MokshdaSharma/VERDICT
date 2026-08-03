import os
from google import genai
from verdict.agents.specialized import EvidenceAgent
from verdict.graph.models import AnyNode, Edge
import json

def test_ea():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Missing API KEY")
        return
        
    client = genai.Client(api_key=api_key)
    agent = EvidenceAgent(api_client=client, model_name="gemini-3.5-flash")
    
    patient_data = """Patient has a temperature of 38.5 C and complains of nausea."""
    print("Running EvidenceAgent...")
    out = agent.run(patient_data)
    print("EA output nodes:")
    for n in out.nodes:
        print(f" - {n.id} : {n.claim}")
        
    print("EA output edges:")
    for e in out.edges:
        print(f" - {e.source_id} -> {e.target_id}")

if __name__ == "__main__":
    test_ea()
