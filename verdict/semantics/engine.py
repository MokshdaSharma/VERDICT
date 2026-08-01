import math
from typing import Dict, List, Set, Tuple

def compute_E(X_sigmas: List[float]) -> float:
    """
    E(X) = 1 - PRODUCT_{x in X} (1 - sigma(x))
    """
    product = 1.0
    for s in X_sigmas:
        product *= (1.0 - s)
    return 1.0 - product

def compute_semantics(
    nodes: List[str],
    w0: Dict[str, float],
    attackers: Dict[str, List[str]],
    supporters: Dict[str, List[str]],
    epsilon: float = 1e-5,
    max_iters: int = 1000
) -> Dict[str, float]:
    """
    Computes the DF-QuAD-style gradual argumentation semantics via fixpoint iteration.
    
    sigma(n) = w0(n) - w0(n) * E(attackers(n)) + (1 - w0(n)) * E(supporters(n))
    """
    # Initialize sigmas with base weights
    sigma = {n: w0.get(n, 0.0) for n in nodes}
    
    for _ in range(max_iters):
        new_sigma = {}
        max_diff = 0.0
        
        for n in nodes:
            base_w = w0.get(n, 0.0)
            
            att_sigmas = [sigma[att] for att in attackers.get(n, [])]
            sup_sigmas = [sigma[sup] for sup in supporters.get(n, [])]
            
            E_att = compute_E(att_sigmas)
            E_sup = compute_E(sup_sigmas)
            
            new_val = base_w - (base_w * E_att) + ((1.0 - base_w) * E_sup)
            # Bound the value between 0 and 1, though mathematically it should naturally stay within if w0 is in [0, 1]
            new_val = max(0.0, min(1.0, new_val))
            
            new_sigma[n] = new_val
            max_diff = max(max_diff, abs(new_val - sigma[n]))
            
        sigma = new_sigma
        
        if max_diff < epsilon:
            break
            
    return sigma
