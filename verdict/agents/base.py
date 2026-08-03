import json
from typing import List, Dict, Any, Type, Optional
from pydantic import BaseModel, Field
from verdict.graph.models import AnyNode, Edge
from verdict.utils.rate_limit import call_with_rate_limit
from google.genai import types

class ProvenanceProp(BaseModel):
    document_id: Optional[str] = None
    rule_id: Optional[str] = None
    source_node_ids: Optional[list[str]] = Field(default_factory=list)
    resource_type: Optional[str] = None
    id: Optional[str] = None
    field: Optional[str] = None

class NodeProp(BaseModel):
    id: str
    type: str
    claim: str
    provenance: ProvenanceProp
    w0: float
    author: str

class EdgeProp(BaseModel):
    source_id: str
    target_id: str
    type: str

class AgentOutput(BaseModel):
    nodes: list[NodeProp] = Field(default_factory=list)
    edges: list[EdgeProp] = Field(default_factory=list)

class BaseAgent:
    def __init__(self, api_client, model_name: str = "gemini-2.5-flash"):
        self.api_client = api_client
        self.model_name = model_name
        
    def _get_system_prompt(self) -> str:
        return "You are an agent."
        
    def run(self, context: str) -> AgentOutput:
        if not self.api_client:
            return AgentOutput(nodes=[], edges=[])
            
        system_prompt = self._get_system_prompt()
        
        # We roughly estimate tokens as characters / 4
        est_tokens = (len(system_prompt) + len(context)) // 4
        print(f"[TOKEN DIAGNOSIS] {self.__class__.__name__} sending ~{est_tokens} input tokens.")
        
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            response_schema=AgentOutput,
        )
        
        try:
            response = call_with_rate_limit(
                self.api_client.models.generate_content,
                est_tokens,
                model=self.model_name,
                contents=context,
                config=config
            )
            
            if response.text:
                return AgentOutput.model_validate_json(response.text)
        except Exception as e:
            print(f"[{self.__class__.__name__}] LLM error: {e}")
            
        return AgentOutput(nodes=[], edges=[])
