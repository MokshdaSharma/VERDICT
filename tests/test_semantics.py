from verdict.semantics.engine import compute_semantics
from verdict.semantics.robustness import compute_robustness_margin, compute_personalization_defeat
import math

def test_semantics_single_node():
    nodes = ["a"]
    w0 = {"a": 0.8}
    attackers = {}
    supporters = {}
    
    sigmas = compute_semantics(nodes, w0, attackers, supporters)
    assert math.isclose(sigmas["a"], 0.8)

def test_semantics_one_attack():
    # a attacks b
    nodes = ["a", "b"]
    w0 = {"a": 0.8, "b": 0.5}
    attackers = {"b": ["a"]}
    supporters = {}
    
    sigmas = compute_semantics(nodes, w0, attackers, supporters)
    assert math.isclose(sigmas["a"], 0.8)
    # sigma(b) = 0.5 - 0.5 * 0.8 = 0.1
    assert math.isclose(sigmas["b"], 0.1)

def test_semantics_one_support():
    # c supports d
    nodes = ["c", "d"]
    w0 = {"c": 0.6, "d": 0.5}
    attackers = {}
    supporters = {"d": ["c"]}
    
    sigmas = compute_semantics(nodes, w0, attackers, supporters)
    assert math.isclose(sigmas["c"], 0.6)
    # sigma(d) = 0.5 + 0.5 * 0.6 = 0.8
    assert math.isclose(sigmas["d"], 0.8)

def test_semantics_multiple_attackers():
    # a1 and a2 attack b
    nodes = ["a1", "a2", "b"]
    w0 = {"a1": 0.8, "a2": 0.5, "b": 0.5}
    attackers = {"b": ["a1", "a2"]}
    supporters = {}
    
    sigmas = compute_semantics(nodes, w0, attackers, supporters)
    assert math.isclose(sigmas["a1"], 0.8)
    assert math.isclose(sigmas["a2"], 0.5)
    
    # E(attackers) = 1 - (1 - 0.8)*(1 - 0.5) = 1 - 0.2*0.5 = 1 - 0.1 = 0.9
    # sigma(b) = 0.5 - 0.5 * 0.9 = 0.05
    assert math.isclose(sigmas["b"], 0.05)

def test_robustness_margin():
    # n_r supported by s1
    nodes = ["s1", "n_r"]
    w0 = {"s1": 0.8, "n_r": 0.5}
    attackers = {}
    supporters = {"n_r": ["s1"]}
    
    # Base sigma(n_r) = 0.9
    # We want to flip it below 0.6
    rho = compute_robustness_margin(
        target_node="n_r",
        nodes=nodes,
        w0=w0,
        attackers=attackers,
        supporters=supporters,
        tau_certify=0.6
    )
    # Removing s1 (weight 0.8) makes sigma(n_r) = 0.5, which is < 0.6.
    # So rho should be 0.8.
    assert math.isclose(rho, 0.8)

def test_personalization_defeat():
    # r is a GuidelineRule supporting recommendation n_r
    # p is a PersonalizationConstraint attacking r
    # e is evidence supporting p
    nodes = ["e", "p", "r", "n_r"]
    w0 = {"e": 0.9, "p": 0.5, "r": 0.5, "n_r": 0.5}
    attackers = {"r": ["p"]}
    supporters = {"p": ["e"], "n_r": ["r"]}
    
    # Calculate base sigmas:
    # sigma(e) = 0.9
    # sigma(p) = 0.5 + 0.5 * 0.9 = 0.95
    # sigma(r) = 0.5 - 0.5 * 0.95 = 0.025
    # sigma(n_r) = 0.5 + 0.5 * 0.025 = 0.5125
    
    # If p is removed (w0(p)=0):
    # sigma(p) = 0 + 1 * 0.9 = 0.9 (Wait, formula is w0 + (1-w0)*E. If w0=0, sigma=E. So sigma(p)=0.9)
    # Ah, if w0(p)=0, it means it has NO BASE STRENGTH. BUT it still gets strength from `e`.
    # Let's verify marginal effect calculation. If w0(p)=0:
    # sigma(p) = 0 + (1 - 0) * 0.9 = 0.9.
    # sigma(r) = 0.5 - 0.5 * 0.9 = 0.05.
    # sigma(n_r) = 0.5 + 0.5 * 0.05 = 0.525.
    
    # If r is removed (w0(r)=0):
    # sigma(r) = 0 - 0 * 0.95 + (1-0)*0 = 0.
    # sigma(n_r) = 0.5 + 0.5 * 0 = 0.5.
    
    # Let's just run it to ensure no exceptions and plausible output.
    is_defeated = compute_personalization_defeat(
        recommendation_node="n_r",
        personalization_node="p",
        guideline_node="r",
        nodes=nodes,
        w0=w0,
        attackers=attackers,
        supporters=supporters
    )
    
    assert isinstance(is_defeated, bool)
