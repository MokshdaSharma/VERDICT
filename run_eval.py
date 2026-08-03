import os
import json
import time
import argparse
from verdict.graph.store import GraphStore
from verdict.verifier.backends import LLMEntailmentBackend
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

import urllib.request
import urllib.error
import subprocess

def check_fhir_server(url="http://localhost:8080/fhir", timeout_s=120):
    print("Checking local FHIR server health...")
    
    def is_up():
        try:
            # MedAgentBench FHIR returns metadata or requires specific endpoint.
            # Just hitting the base with a short timeout works to check port.
            urllib.request.urlopen(url, timeout=2)
            return True
        except urllib.error.URLError as e:
            # 404 or 401 is fine, it means the server is running and responding
            if hasattr(e, 'code'):
                return True
            return False
        except Exception:
            return False

    if is_up():
        print("FHIR server is already running.")
        return

    print("FHIR server is down. Attempting to start via docker compose...")
    try:
        subprocess.run(["docker", "compose", "up", "-d"], check=True)
    except Exception as e:
        raise RuntimeError(f"Failed to start docker container: {e}")
        
    print("Waiting for server to become ready...")
    start_time = time.time()
    while time.time() - start_time < timeout_s:
        if is_up():
            print("Server is up and ready.")
            return
        time.sleep(2)
        
    raise RuntimeError("Timed out waiting for local FHIR server to start.")

def setup_orchestrator(api_client=None, bulk_model="gemini-2.5-flash", synth_model="gemini-3.5-flash"):
    # Initialize components
    graph_store = GraphStore()
    
    # Use LLM Entailment backend due to local PyArrow crash
    from verdict.verifier.backends import LLMEntailmentBackend
    verifier = LLMEntailmentBackend(api_client, bulk_model)
    admission_gate = AdmissionGate(verifier=verifier, tau_entail=0.5)
    
    agents = {
        "EA": EvidenceAgent(api_client=api_client, model_name=bulk_model),
        "KA": KnowledgeAgent(api_client=api_client, model_name=bulk_model),
        "GA": GuidelineAgent(api_client=api_client, model_name=bulk_model),
        "HA": HypothesisAgent(api_client=api_client, model_name=synth_model),
        "PA": PersonalizationAgent(api_client=api_client, model_name=bulk_model),
        "ACA": AdversarialCriticAgent(api_client=api_client, model_name=bulk_model)
    }
    
    return Orchestrator(graph_store, admission_gate, agents)

def main():
    parser = argparse.ArgumentParser(description="Run VERDICT Evaluation")
    parser.add_argument("--bulk-model", type=str, default="gemini-2.5-flash", help="Cheaper/faster model for bulk extraction (EA/KA/GA/PA/ACA)")
    parser.add_argument("--synth-model", type=str, default="gemini-3.5-flash", help="Stronger model for synthesis (HA)")
    parser.add_argument("--limit", type=int, default=1, help="Max number of patients to evaluate per dataset")
    parser.add_argument("--force-rerun", action="store_true", help="Rerun and overwrite existing results.jsonl")
    args = parser.parse_args()

    check_fhir_server("http://localhost:8080/fhir")

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
        from google import genai
        # We always use the native GenAI SDK now.
        api_client = genai.Client(api_key=api_key)
        
        # 2c. Validate configuration at startup
        print("Validating model availability...")
        try:
            available_models = [m.name for m in api_client.models.list()]
            if f"models/{args.bulk_model}" not in available_models and args.bulk_model not in available_models:
                raise ValueError(f"Configured bulk-model '{args.bulk_model}' not found in available models.")
            if f"models/{args.synth_model}" not in available_models and args.synth_model not in available_models:
                raise ValueError(f"Configured synth-model '{args.synth_model}' not found in available models.")
            print(f"Using Google GenAI API with models: {args.bulk_model} / {args.synth_model}")
        except Exception as e:
            if isinstance(e, ValueError): raise e
            print(f"Warning: Model list verification failed: {e}")
    else:
        print("No OPENAI_API_KEY found in environment. Using mocked LLM fallback.")
        
    results = {}
    # 5 evaluation arms
    arms = [
        "single-agent+RAG",
        "free-text multi-agent",
        "VERDICT-verifier-ablated",
        "VERDICT-fixed-scheduling",
        "full VERDICT"
    ]
    
    def get_orchestrator_fn(arm_name):
        # We need fresh components per run
        graph_store = GraphStore()
        
        # Determine if we should ablate the verifier
        if arm_name == "VERDICT-verifier-ablated":
            # Pass tau_entail = -1.0 or None to bypass
            from verdict.verifier.backends import LLMEntailmentBackend
            verifier = LLMEntailmentBackend(api_client, args.bulk_model)
            admission_gate = AdmissionGate(verifier=verifier, tau_entail=-1.0)
        else:
            from verdict.verifier.backends import LLMEntailmentBackend
            verifier = LLMEntailmentBackend(api_client, args.bulk_model)
            admission_gate = AdmissionGate(verifier=verifier, tau_entail=0.5)
            
        agents = {
            "EA": EvidenceAgent(api_client=api_client, model_name=args.bulk_model),
            "KA": KnowledgeAgent(api_client=api_client, model_name=args.bulk_model),
            "GA": GuidelineAgent(api_client=api_client, model_name=args.bulk_model),
            "HA": HypothesisAgent(api_client=api_client, model_name=args.synth_model),
            "PA": PersonalizationAgent(api_client=api_client, model_name=args.bulk_model),
            "ACA": AdversarialCriticAgent(api_client=api_client, model_name=args.bulk_model)
        }
        
        return Orchestrator(graph_store, admission_gate, agents)

    # Helper to evaluate
    def eval_dataset(dataset_name, loader_or_client, ids):
        if not ids:
            return
        eval_ids = ids[:args.limit]
        
        for arm in arms:
            print(f"\n--- Running {dataset_name} on {arm} ---")
            res = run_evaluation(
                loader_or_client, 
                get_orchestrator_fn, 
                eval_ids, 
                arm_name=arm,
                force_rerun=args.force_rerun,
                api_client=api_client,
                model_name=args.bulk_model
            )
            
            if dataset_name not in results:
                results[dataset_name] = {}
            results[dataset_name][arm] = res

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
