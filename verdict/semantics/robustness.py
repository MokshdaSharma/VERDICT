from typing import Dict, List, Set, Tuple
from verdict.semantics.engine import compute_semantics

def marginal_effect(
    target_node: str,
    node_to_remove: str,
    nodes: List[str],
    w0: Dict[str, float],
    attackers: Dict[str, List[str]],
    supporters: Dict[str, List[str]],
    current_sigma: float
) -> float:
    """
    Computes the marginal effect on target_node's sigma if node_to_remove's base strength is zeroed out.
    """
    modified_w0 = w0.copy()
    modified_w0[node_to_remove] = 0.0
    
    new_sigmas = compute_semantics(nodes, modified_w0, attackers, supporters)
    # The effect is how much the target node's sigma DECREASES.
    # If the target is a recommendation, we assume we want to know what reduces its strength.
    return current_sigma - new_sigmas.get(target_node, 0.0)


def compute_robustness_margin(
    target_node: str,
    nodes: List[str],
    w0: Dict[str, float],
    attackers: Dict[str, List[str]],
    supporters: Dict[str, List[str]],
    tau_certify: float
) -> float:
    """
    Computes rho(n_r): iteratively zero out the base strength of the node with the highest 
    marginal effect on sigma(target_node) until certification flips (sigma < tau_certify).
    """
    current_w0 = w0.copy()
    cumulative_removed_weight = 0.0
    
    while True:
        sigmas = compute_semantics(nodes, current_w0, attackers, supporters)
        current_target_sigma = sigmas.get(target_node, 0.0)
        
        if current_target_sigma < tau_certify:
            break
            
        best_node = None
        max_effect = -float('inf')
        
        for n in nodes:
            if current_w0.get(n, 0.0) == 0.0:
                continue
                
            effect = marginal_effect(target_node, n, nodes, current_w0, attackers, supporters, current_target_sigma)
            
            if effect > max_effect:
                max_effect = effect
                best_node = n
                
        if best_node is None or max_effect <= 0:
            # Cannot reduce the sigma any further
            break
            
        # Zero out the best node
        cumulative_removed_weight += current_w0[best_node]
        current_w0[best_node] = 0.0
        
    return cumulative_removed_weight


def compute_personalization_defeat(
    recommendation_node: str,
    personalization_node: str,
    guideline_node: str,
    nodes: List[str],
    w0: Dict[str, float],
    attackers: Dict[str, List[str]],
    supporters: Dict[str, List[str]]
) -> bool:
    """
    Computes if a PersonalizationConstraint (p) successfully defeats a GuidelineRule (r).
    Returns True if delta(p) > delta(r), where delta is the absolute marginal contribution to sigma(n_r).
    """
    sigmas = compute_semantics(nodes, w0, attackers, supporters)
    current_target_sigma = sigmas.get(recommendation_node, 0.0)
    
    # marginal_effect calculates (current - new), so we take absolute value to measure magnitude of contribution
    delta_p = abs(marginal_effect(recommendation_node, personalization_node, nodes, w0, attackers, supporters, current_target_sigma))
    delta_r = abs(marginal_effect(recommendation_node, guideline_node, nodes, w0, attackers, supporters, current_target_sigma))
    
    return delta_p > delta_r
