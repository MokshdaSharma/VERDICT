from typing import List
from verdict.orchestrator.loop import Orchestrator

def run_evaluation(dataset_adapter, orchestrator: Orchestrator, patient_ids: List[str]):
    results = []
    
    for pid in patient_ids:
        try:
            patient = dataset_adapter.get_patient(pid)
            data_str = f"Demographics: {patient.get_demographics()}\nObservations: {patient.get_observations()}"
            
            orchestrator.run_case(data_str, max_critic_rounds=1)
            
            # Extract final recommendation
            from verdict.orchestrator.decision import DecisionLayer
            decision_layer = DecisionLayer()
            
            recs = [n for n in orchestrator.graph_store.get_all_nodes() if n.type == "Recommendation"]
            if recs:
                best_rec = max(recs, key=lambda x: x.w0)
                eval_res = decision_layer.evaluate(orchestrator.graph_store, best_rec.id)
                results.append({"patient_id": pid, "status": "Success", "eval": eval_res})
            else:
                results.append({"patient_id": pid, "status": "No Recommendation"})
        except Exception as e:
            results.append({"patient_id": pid, "status": "Error", "error": str(e)})
            
    return results

if __name__ == "__main__":
    # Example usage hook
    pass
