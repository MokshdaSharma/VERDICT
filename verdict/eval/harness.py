import json
import os
import copy
from typing import List
from verdict.orchestrator.loop import Orchestrator
from verdict.eval.baseline import BaselineAgent
from verdict.graph.store import GraphStore

class MultiAgentBaseline:
    def __init__(self, api_client, model_name="gemini-2.5-flash"):
        self.api_client = api_client
        self.model_name = model_name

    def evaluate(self, patient_data: str) -> str:
        if not self.api_client:
            return "Mock Multi-Agent Recommendation: Administer fluids."
            
        from google.genai import types
        from verdict.utils.rate_limit import call_with_rate_limit
        
        # Agent 1: Summarize Evidence
        prompt1 = f"Extract key clinical evidence from: {patient_data}"
        resp1 = call_with_rate_limit(self.api_client.models.generate_content, len(prompt1)//4, model=self.model_name, contents=prompt1)
        evidence = resp1.text
        
        # Agent 2: Propose Recommendation
        prompt2 = f"Based on this evidence, propose a clinical recommendation:\n{evidence}"
        resp2 = call_with_rate_limit(self.api_client.models.generate_content, len(prompt2)//4, model=self.model_name, contents=prompt2)
        rec = resp2.text
        
        # Agent 3: Critic
        prompt3 = f"Critique this recommendation based on the evidence. Is there any contraindication?\nEvidence: {evidence}\nRecommendation: {rec}\nProvide final recommendation."
        resp3 = call_with_rate_limit(self.api_client.models.generate_content, len(prompt3)//4, model=self.model_name, contents=prompt3)
        return resp3.text

def run_evaluation(dataset_adapter, get_orchestrator_fn, patient_ids: List[str], arm_name: str, force_rerun: bool = False, api_client=None, model_name="gemini-2.5-flash"):
    results = []
    
    log_file = os.path.join("logs", f"eval_results_{arm_name}.jsonl")
    completed_pids = set()
    if not force_rerun and os.path.exists(log_file):
        with open(log_file, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    completed_pids.add(data.get("patient_id"))
                except: pass

    for pid in patient_ids:
        if pid in completed_pids:
            print(f"Skipping {pid} for {arm_name}, already evaluated.")
            # Read cached to return
            with open(log_file, "r") as f:
                for line in f:
                    data = json.loads(line)
                    if data.get("patient_id") == pid:
                        results.append(data)
            continue

        print(f"Evaluating {pid} for {arm_name}...")
        try:
            patient = dataset_adapter.get_patient(pid)
            data_str = f"Demographics: {patient.get_demographics()}\nObservations: {patient.get_observations()}\nConditions: {patient.get_conditions()}"
            
            res_dict = {"patient_id": pid, "status": "Success", "arm": arm_name}
            
            if arm_name == "single-agent+RAG":
                agent = BaselineAgent(api_client, model_name)
                # Mock RAG by appending a static guideline text
                guidelines = "General Guidelines: Treat infections with antibiotics. Adjust for renal impairment."
                rec = agent.evaluate(data_str, guidelines)
                res_dict["recommendation"] = rec
                
            elif arm_name == "free-text multi-agent":
                agent = MultiAgentBaseline(api_client, model_name)
                rec = agent.evaluate(data_str)
                res_dict["recommendation"] = rec
                
            else:
                # VERDICT architectures
                orchestrator = get_orchestrator_fn(arm_name)
                
                # If full VERDICT, we use ACA-priority scheduler (mocked as running ACA more dynamically)
                # If fixed-scheduling, we use the standard linear loop (max_critic_rounds=1)
                is_fixed = (arm_name == "VERDICT-fixed-scheduling")
                orchestrator.run_case(data_str, max_critic_rounds=1 if is_fixed else 3)
                
                from verdict.orchestrator.decision import DecisionLayer
                decision_layer = DecisionLayer()
                recs = [n for n in orchestrator.graph_store.get_all_nodes() if n.type == "Recommendation"]
                
                if recs:
                    best_rec = max(recs, key=lambda x: x.w0)
                    eval_res = decision_layer.evaluate(orchestrator.graph_store, best_rec.id)
                    res_dict["eval"] = eval_res
                    res_dict["recommendation"] = best_rec.description
                    
                    # Compute Faithfulness (Counterfactual test)
                    # Pick an observation node supporting the recommendation
                    obs_nodes = [n for n in orchestrator.graph_store.get_all_nodes() if n.type == "Observation"]
                    if obs_nodes:
                        # Test counterfactual
                        obs_to_remove = obs_nodes[0]
                        
                        # We must deeply copy the graph store state
                        cf_store = GraphStore()
                        cf_store.nodes = copy.deepcopy(orchestrator.graph_store.nodes)
                        cf_store.edges = copy.deepcopy(orchestrator.graph_store.edges)
                        cf_store._nx_graph = orchestrator.graph_store._nx_graph.copy()
                        
                        cf_store._nx_graph.remove_node(obs_to_remove.id)
                        del cf_store.nodes[obs_to_remove.id]
                        
                        # Re-evaluate
                        cf_eval_res = decision_layer.evaluate(cf_store, best_rec.id)
                        
                        explanation_changed = (cf_eval_res != eval_res)
                        res_dict["faithfulness_changed"] = explanation_changed
                    
                    # Compute Ungrounded Claim Rate
                    # Count claims (nodes) with valid provenance
                    total_claims = len(orchestrator.graph_store.nodes)
                    grounded = sum(1 for n in orchestrator.graph_store.get_all_nodes() if getattr(n, 'provenance', None) and len(n.provenance) > 0)
                    res_dict["ungrounded_rate"] = (total_claims - grounded) / total_claims if total_claims > 0 else 0
                    
                else:
                    res_dict["status"] = "No Recommendation"
                    
            results.append(res_dict)
            with open(log_file, "a") as f:
                f.write(json.dumps(res_dict) + "\n")
                
        except Exception as e:
            res_dict = {"patient_id": pid, "status": "Error", "error": str(e), "arm": arm_name}
            results.append(res_dict)
            with open(log_file, "a") as f:
                f.write(json.dumps(res_dict) + "\n")
            
    return results
