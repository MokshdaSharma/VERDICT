from typing import Dict, Any, Optional
from verdict.graph.store import GraphStore
from verdict.semantics.engine import compute_semantics
from verdict.semantics.robustness import compute_robustness_margin

class DecisionLayer:
    def __init__(self, tau_certify: float = 0.5, tau_robust: float = 0.2):
        self.tau_certify = tau_certify
        self.tau_robust = tau_robust
        
    def evaluate(self, graph_store: GraphStore, target_node_id: str) -> Dict[str, Any]:
        nodes = [n.id for n in graph_store.get_all_nodes()]
        w0 = {n.id: n.w0 for n in graph_store.get_all_nodes()}
        
        attackers = {n: graph_store.get_attackers(n) for n in nodes}
        supporters = {n: graph_store.get_supporters(n) for n in nodes}
        
        sigmas = compute_semantics(nodes, w0, attackers, supporters)
        sigma_nr = sigmas.get(target_node_id, 0.0)
        
        if sigma_nr < self.tau_certify:
            return {
                "decision": "NO-RECOMMENDATION",
                "sigma": sigma_nr,
                "rho": 0.0,
                "reason": "Strength below threshold."
            }
            
        rho_nr = compute_robustness_margin(target_node_id, nodes, w0, attackers, supporters, self.tau_certify)
        
        if rho_nr < self.tau_robust:
            return {
                "decision": "CERTIFIED-MUST-REVIEW",
                "sigma": sigma_nr,
                "rho": rho_nr,
                "reason": "Strength ok, but robustness too low."
            }
            
        return {
            "decision": "CERTIFIED",
            "sigma": sigma_nr,
            "rho": rho_nr,
            "reason": "Autonomous-eligible."
        }
