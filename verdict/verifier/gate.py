from typing import Dict, Any, Tuple
from verdict.graph.models import AnyNode
from verdict.verifier.backends import EntailmentVerifier

class AdmissionGate:
    def __init__(self, verifier: EntailmentVerifier, tau_entail: float = 0.5):
        self.verifier = verifier
        self.tau_entail = tau_entail
        self.rejection_log = []
        
    def check_node(self, node: AnyNode, provenance_text: str) -> Tuple[bool, float]:
        """
        Checks if the node's claim is entailed by the provenance text.
        Updates node.w0 if accepted.
        Returns (is_accepted, w0_score).
        """
        score = self.verifier.verify(premise=provenance_text, hypothesis=node.claim)
        
        if score < self.tau_entail:
            self.rejection_log.append({
                "node_id": node.id,
                "claim": node.claim,
                "score": score,
                "reason": f"Score {score} < threshold {self.tau_entail}"
            })
            return False, score
            
        node.w0 = score
        return True, score
