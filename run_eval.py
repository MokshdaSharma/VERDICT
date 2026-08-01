import os
import json
import time
import argparse
from verdict.graph.store import GraphStore
from verdict.verifier.backends import HuggingFaceNLIBackend, LLMEntailmentBackend
from verdict.verifier.gate import AdmissionGate
from verdict.agents.specialized import (
    EvidenceAgent, KnowledgeAgent, GuidelineAgent,
    HypothesisAgent, PersonalizationAgent, AdversarialCriticAgent
)
from verdict.orchestrator.loop import Orchestrator
from verdict.orchestrator.renderer import Renderer
from verdict.eval.harness import run_evaluation
from verdict.datasets.medagentbench import MedAgentBenchClient
from verdict.datasets.mimic import MimicFhirLoader
from verdict.datasets.synthea import SyntheaLoader

def setup_orchestrator(api_client=None, model_name="gemini-3.5-flash"):
    # Initialize components
    graph_store = GraphStore()
    
    # We will use the LLM backend for speed/simplicity in this eval if api_client is available
    # Otherwise fallback to the mock.
    verifier = LLMEntailmentBackend(api_client=api_client, model_name=model_name)
    admission_gate = AdmissionGate(verifier=verifier, tau_entail=0.5)
    
    agents = {
        "EA": EvidenceAgent(api_client=api_client, model_name=model_name),
        "KA": KnowledgeAgent(api_client=api_client, model_name=model_name),
        "GA": GuidelineAgent(api_client=api_client, model_name=model_name),
        "HA": HypothesisAgent(api_client=api_client, model_name=model_name),
        "PA": PersonalizationAgent(api_client=api_client, model_name=model_name),
        "ACA": AdversarialCriticAgent(api_client=api_client, model_name=model_name)
    }
    
    return Orchestrator(graph_store, admission_gate, agents)

def main():
    parser = argparse.ArgumentParser(description="Run VERDICT Evaluation")
    parser.add_argument("--model", type=str, default="gemini-3.5-flash", help="Model name to use with API")
    parser.add_argument("--sleep", type=int, default=0, help="Sleep time in seconds between patient evaluations (useful for free tier limits)")
    parser.add_argument("--limit", type=int, default=1, help="Max number of patients to evaluate per dataset")
    args = parser.parse_args()

    print("Initializing datasets...")
    # Initialize Datasets
    mab_client = MedAgentBenchClient("http://localhost:8080/fhir")
    
    mimic_dir = os.path.join("data", "mimic", "bundles")
    mimic_loader = MimicFhirLoader(mimic_dir)
    
    synthea_dir = os.path.join("data", "synthea", "output", "fhir")
    synthea_loader = SyntheaLoader(synthea_dir)
    
    print("Setting up Orchestrator...")
    
    # Initialize API client if key exists in environment
    api_key = os.environ.get("OPENAI_API_KEY")
    api_client = None
    if api_key:
        import openai
        # Check if it's a Gemini key by prefix or length
        if api_key.startswith("AIza") or api_key.startswith("AQ."):
            api_client = openai.OpenAI(
                api_key=api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
            )
            print(f"Using Gemini API via OpenAI compatible endpoint with model: {args.model}")
        else:
            api_client = openai.OpenAI(api_key=api_key)
            print(f"Using OpenAI API with model: {args.model}")
    else:
        print("No OPENAI_API_KEY found in environment. Using mocked LLM fallback.")
        
    results = {}
    
    # Helper to evaluate
    def eval_dataset(dataset_name, loader_or_client, ids):
        if not ids:
            return
        eval_ids = ids[:args.limit]
        print(f"Testing {dataset_name} on patients: {eval_ids}...")
        orchestrator = setup_orchestrator(api_client, model_name=args.model)
        res = run_evaluation(loader_or_client, orchestrator, eval_ids)
        results[dataset_name] = res
        if args.sleep > 0:
            print(f"Sleeping for {args.sleep}s to reset API token quotas...")
            time.sleep(args.sleep)

    # Run evaluations
    eval_dataset("MedAgentBench", mab_client, mab_client.get_all_patient_ids())
    eval_dataset("MIMIC", mimic_loader, mimic_loader.get_all_patient_ids())
    eval_dataset("Synthea", synthea_loader, synthea_loader.get_all_patient_ids())
        
    # Write to RESULTS.md
    with open("RESULTS.md", "a") as f:
        f.write("\n## Evaluation Run Results\n")
        f.write("```json\n")
        f.write(json.dumps(results, indent=2))
        f.write("\n```\n")
        
    print("Done. Results appended to RESULTS.md.")

if __name__ == "__main__":
    main()
