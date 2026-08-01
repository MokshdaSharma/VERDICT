from verdict.graph.store import GraphStore

class Renderer:
    def __init__(self, graph_store: GraphStore):
        self.graph_store = graph_store
        
    def render_tree(self, target_node_id: str, depth: int = 0, visited=None) -> str:
        """
        Recursively renders an indented text tree of arguments for a target node.
        """
        if visited is None:
            visited = set()
            
        if target_node_id in visited:
            return "  " * depth + f"- {target_node_id} (cycle detected)"
            
        visited.add(target_node_id)
        node = self.graph_store.get_node(target_node_id)
        if not node:
            return "  " * depth + f"- {target_node_id} (not found)"
            
        result = "  " * depth + f"- [{node.type}] {node.id}: {node.claim} (w0={node.w0:.2f})\n"
        
        supporters = self.graph_store.get_supporters(target_node_id)
        for sup in supporters:
            result += "  " * (depth + 1) + f"SUPPORTS:\n"
            result += self.render_tree(sup, depth + 2, visited.copy()) + "\n"
            
        attackers = self.graph_store.get_attackers(target_node_id)
        for att in attackers:
            result += "  " * (depth + 1) + f"ATTACKS:\n"
            result += self.render_tree(att, depth + 2, visited.copy()) + "\n"
            
        return result.strip()
