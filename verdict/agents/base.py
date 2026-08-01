import json
from typing import List, Dict, Any, Type, Optional
from pydantic import BaseModel
from verdict.graph.models import AnyNode, Edge

class AgentOutput(BaseModel):
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]

class BaseAgent:
    def __init__(self, api_client, model_name: str = "gpt-4o-mini"):
        self.api_client = api_client
        self.model_name = model_name
        
    def _get_system_prompt(self) -> str:
        return "You are an agent."
        
    def _get_tool_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "propose_graph_updates",
                "description": "Propose nodes and edges to add to the Clinical Argument Graph.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "nodes": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string"},
                                    "type": {"type": "string"},
                                    "claim": {"type": "string"},
                                    "provenance": {"type": "object"},
                                    "w0": {"type": "number"},
                                    "author": {"type": "string"}
                                },
                                "required": ["id", "type", "claim", "provenance", "w0", "author"]
                            }
                        },
                        "edges": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "source_id": {"type": "string"},
                                    "target_id": {"type": "string"},
                                    "type": {"type": "string"}
                                },
                                "required": ["source_id", "target_id", "type"]
                            }
                        }
                    },
                    "required": ["nodes", "edges"]
                }
            }
        }
        
    def run(self, context: str) -> AgentOutput:
        if not self.api_client:
            # Fallback for testing when no client is provided
            return AgentOutput(nodes=[], edges=[])
            
        system_prompt = self._get_system_prompt()
        tool = self._get_tool_schema()
        
        response = self.api_client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context}
            ],
            tools=[tool],
            tool_choice={"type": "function", "function": {"name": "propose_graph_updates"}}
        )
        
        choice = response.choices[0]
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                if tc.function.name == "propose_graph_updates":
                    args = json.loads(tc.function.arguments)
                    return AgentOutput(
                        nodes=args.get("nodes", []),
                        edges=args.get("edges", [])
                    )
                    
        return AgentOutput(nodes=[], edges=[])
