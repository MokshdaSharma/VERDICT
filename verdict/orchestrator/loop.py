from typing import Dict, Any, List, Optional
from verdict.graph.store import GraphStore
from verdict.graph.models import Edge
from verdict.verifier.gate import AdmissionGate

class Orchestrator:
    def __init__(
        self,
        graph_store: GraphStore,
        admission_gate: AdmissionGate,
        agents: Dict[str, Any]
    ):
        self.graph_store = graph_store
        self.admission_gate = admission_gate
        self.agents = agents
        self.logs = []
        
    def _run_agent(self, agent_key: str, context: str):
        agent = self.agents.get(agent_key)
        if not agent:
            return
            
        output = agent.run(context)
        
        # Process Nodes
        for node_data in output.nodes:
            # We import here to avoid circular dependencies if any
            from pydantic import TypeAdapter
            from verdict.graph.models import AnyNode
            
            node_adapter = TypeAdapter(AnyNode)
            try:
                if hasattr(node_data, "model_dump"):
                    node_data = node_data.model_dump()
                node = node_adapter.validate_python(node_data)
                
                # Check entailment via Grounding Verifier
                # In a real system, provenance_text is fetched dynamically.
                # Here we just pass the claim itself for the mock or a dummy string.
                provenance_text = "Dummy provenance text for POC" 
                is_accepted, score = self.admission_gate.check_node(node, provenance_text)
                
                if is_accepted:
                    self.graph_store.add_node(node)
                    self.logs.append(f"[{agent_key}] Admitted node {node.id} with score {score}")
                else:
                    self.logs.append(f"[{agent_key}] Rejected node {node.id}. Score: {score}")
            except Exception as e:
                self.logs.append(f"[{agent_key}] Node validation error: {e}")
                
        # Process Edges
        for edge_data in output.edges:
            try:
                if hasattr(edge_data, "model_dump"):
                    edge_data = edge_data.model_dump()
                edge = Edge(**edge_data)
                self.graph_store.add_edge(edge)
                self.logs.append(f"[{agent_key}] Added edge {edge.source_id} -> {edge.target_id}")
            except Exception as e:
                self.logs.append(f"[{agent_key}] Edge validation error: {e}")

    def run_case(self, patient_data: str, max_critic_rounds: int = 2):
        self.logs.append("Starting case...")
        
        # EA -> KA/GA -> HA -> PA
        self._run_agent("EA", f"Patient Data:\n{patient_data}")
        
        graph_context = self.graph_store.to_json()
        self._run_agent("KA", f"Current Graph:\n{graph_context}")
        self._run_agent("GA", f"Current Graph:\n{graph_context}")
        
        graph_context = self.graph_store.to_json()
        self._run_agent("HA", f"Current Graph:\n{graph_context}")
        
        graph_context = self.graph_store.to_json()
        self._run_agent("PA", f"Current Graph:\n{graph_context}")
        
        # ACA loop
        for i in range(max_critic_rounds):
            graph_context = self.graph_store.to_json()
            # Stop if no recommendation exists
            has_rec = any(n.type == "Recommendation" for n in self.graph_store.get_all_nodes())
            if not has_rec:
                self.logs.append("No Recommendation to attack. Ending critic loop.")
                break
                
            pre_edge_count = len(self.graph_store._nx_graph.edges)
            self._run_agent("ACA", f"Current Graph:\n{graph_context}")
            post_edge_count = len(self.graph_store._nx_graph.edges)
            
            if post_edge_count == pre_edge_count:
                self.logs.append("ACA found no new attacks. Ending critic loop.")
                break
                
        self.logs.append("Case complete.")
