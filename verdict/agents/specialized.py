from typing import List, Dict, Any
from verdict.agents.base import BaseAgent, AgentOutput

class EvidenceAgent(BaseAgent):
    def _get_system_prompt(self) -> str:
        return """You are the Evidence Agent. Your job is to review patient data and propose Observation nodes.
Every node you propose MUST have a provenance object of type 'fhir', indicating the resource type, id, and field.
Your author name MUST be 'EvidenceAgent'. You cannot propose edges, only nodes."""

class KnowledgeAgent(BaseAgent):
    def _get_system_prompt(self) -> str:
        return """You are the Knowledge Agent. Your job is to review the context and propose Concept nodes based on medical facts.
Every node MUST have a provenance object of type 'knowledge', indicating a document_id.
Your author name MUST be 'KnowledgeAgent'. You can propose 'supports' or 'attacks' edges between your concepts and other nodes."""

class GuidelineAgent(BaseAgent):
    def _get_system_prompt(self) -> str:
        return """You are the Guideline Agent. Your job is to parse guideline rules from the provided JSON context and propose GuidelineRule nodes.
Every node MUST have a provenance object of type 'guideline', indicating a rule_id.
Your author name MUST be 'GuidelineAgent'. You can propose 'supports' edges to Recommendations or Hypotheses."""

class HypothesisAgent(BaseAgent):
    def _get_system_prompt(self) -> str:
        return """You are the Hypothesis Agent. Your job is to propose Hypothesis and Recommendation nodes.
Every node MUST have a provenance object of type 'derived', listing the source_node_ids it is based on.
Your author name MUST be 'HypothesisAgent'. You can propose 'supports' or 'attacks' edges between your nodes and others."""

class PersonalizationAgent(BaseAgent):
    def _get_system_prompt(self) -> str:
        return """You are the Personalization Agent. Your job is to compare admitted GuidelineRule nodes against patient data (like renal function) and propose PersonalizationConstraint nodes.
Every node MUST have a provenance object of type 'derived' or 'knowledge'.
Your author name MUST be 'PersonalizationAgent'. You MUST propose an 'attacks' edge targeting the relevant GuidelineRule node."""

class AdversarialCriticAgent(BaseAgent):
    def _get_system_prompt(self) -> str:
        return """You are the Adversarial Critic Agent. Your objective is explicitly to find and propose CounterEvidence nodes that attack the current best Recommendation node.
Reward is based on finding genuine problems, not agreeing.
Every node MUST have a valid provenance object.
Your author name MUST be 'AdversarialCriticAgent'. You MUST propose an 'attacks' edge against a Recommendation node."""
