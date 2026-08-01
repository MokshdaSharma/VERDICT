import json
import networkx as nx
from typing import Dict, List, Optional
from verdict.graph.models import AnyNode, Edge

class GraphStore:
    """
    In-memory graph store wrapping NetworkX with JSON persistence.
    """
    def __init__(self):
        self._nx_graph = nx.DiGraph()
        self._nodes: Dict[str, AnyNode] = {}
        
    def add_node(self, node: AnyNode):
        """Adds a node to the graph if it doesn't already exist."""
        if node.id not in self._nodes:
            self._nodes[node.id] = node
            self._nx_graph.add_node(node.id, **node.model_dump())
            
    def add_edge(self, edge: Edge):
        """Adds an edge to the graph. Raises error if nodes are not present."""
        if edge.source_id not in self._nodes:
            raise ValueError(f"Source node {edge.source_id} not in graph.")
        if edge.target_id not in self._nodes:
            raise ValueError(f"Target node {edge.target_id} not in graph.")
            
        self._nx_graph.add_edge(edge.source_id, edge.target_id, type=edge.type)
        
    def get_node(self, node_id: str) -> Optional[AnyNode]:
        return self._nodes.get(node_id)
        
    def get_all_nodes(self) -> List[AnyNode]:
        return list(self._nodes.values())
        
    def get_attackers(self, target_id: str) -> List[str]:
        attackers = []
        if target_id in self._nx_graph:
            for pred in self._nx_graph.predecessors(target_id):
                if self._nx_graph.edges[pred, target_id]['type'] == 'attacks':
                    attackers.append(pred)
        return attackers
        
    def get_supporters(self, target_id: str) -> List[str]:
        supporters = []
        if target_id in self._nx_graph:
            for pred in self._nx_graph.predecessors(target_id):
                if self._nx_graph.edges[pred, target_id]['type'] == 'supports':
                    supporters.append(pred)
        return supporters
        
    def to_json(self) -> str:
        data = {
            "nodes": [node.model_dump() for node in self._nodes.values()],
            "edges": [
                {"source_id": u, "target_id": v, "type": data["type"]}
                for u, v, data in self._nx_graph.edges(data=True)
            ]
        }
        return json.dumps(data, indent=2)
        
    @classmethod
    def from_json(cls, json_str: str) -> "GraphStore":
        # Import dynamically if needed, but we can rely on Pydantic's TypeAdapter for parsing.
        from pydantic import TypeAdapter
        node_adapter = TypeAdapter(AnyNode)
        edge_adapter = TypeAdapter(Edge)
        
        data = json.loads(json_str)
        store = cls()
        
        for n_dict in data.get("nodes", []):
            node = node_adapter.validate_python(n_dict)
            store.add_node(node)
            
        for e_dict in data.get("edges", []):
            edge = edge_adapter.validate_python(e_dict)
            store.add_edge(edge)
            
        return store
