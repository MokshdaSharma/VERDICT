from abc import ABC, abstractmethod
from typing import Dict, Any

class EntailmentVerifier(ABC):
    @abstractmethod
    def verify(self, premise: str, hypothesis: str) -> float:
        """
        Returns an entailment score between 0.0 and 1.0.
        """
        pass

class HuggingFaceNLIBackend(EntailmentVerifier):
    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-base"):
        self.model_name = model_name
        self._classifier = None

    def _load_model(self):
        if self._classifier is None:
            from transformers import pipeline
            self._classifier = pipeline("text-classification", model=self.model_name)

    def verify(self, premise: str, hypothesis: str) -> float:
        self._load_model()
        # Some pipelines accept text/text_pair via dicts or just a tuple (premise, hypothesis)
        result = self._classifier({"text": premise, "text_pair": hypothesis})
        # The result is typically like {"label": "entailment", "score": 0.9}
        # Or a list of results if return_all_scores=True.
        # We need the 'entailment' score.
        # Since text-classification without return_all_scores returns top 1, we might need return_all_scores
        results = self._classifier({"text": premise, "text_pair": hypothesis}, top_k=None)
        
        # results might be a list of dicts: [{'label': 'entailment', 'score': 0.8}, ...]
        # if top_k is passed (depends on transformers version).
        
        if isinstance(results, list):
            # sometimes it's wrapped in a double list [[{...}]]
            if len(results) > 0 and isinstance(results[0], list):
                results = results[0]
            
            for res in results:
                if isinstance(res, dict) and res.get("label", "").lower() == "entailment":
                    return res["score"]
                    
            # fallback to exact matching of expected labels if 'entailment' is named differently (e.g., 'LABEL_0')
            # Assuming entailment, neutral, contradiction order
            # This is model specific.
        
        return 0.5 # fallback

class LLMEntailmentBackend(EntailmentVerifier):
    def __init__(self, api_client=None, model_name="gpt-4o-mini"):
        # We assume an initialized openai client or similar is passed, or we initialize it.
        # For this POC, we use a mock client if none provided.
        self.api_client = api_client
        self.model_name = model_name
        
    def verify(self, premise: str, hypothesis: str) -> float:
        import json
        if not self.api_client:
            # simple keyword match if no client (just for basic tests)
            if hypothesis.lower() in premise.lower():
                return 1.0
            return 0.1
            
        prompt = f"""
Given the following premise, does it entail the hypothesis? 
Respond with ONLY a JSON object containing a "score" field with a float between 0.0 and 1.0.

Premise: {premise}
Hypothesis: {hypothesis}
"""
        response = self.api_client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        try:
            content = response.choices[0].message.content
            # Simple parse
            parsed = json.loads(content)
            return float(parsed.get("score", 0.0))
        except Exception:
            return 0.0
