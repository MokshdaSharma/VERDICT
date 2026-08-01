class BaselineAgent:
    """
    A direct LLM baseline for comparison against the VERDICT architecture.
    """
    def __init__(self, api_client, model_name: str = "claude-3-haiku-20240307"):
        self.api_client = api_client
        self.model_name = model_name
        
    def evaluate(self, patient_data: str, guidelines: str) -> str:
        if not self.api_client:
            return "Mock Baseline Recommendation: Administer fluids."
            
        prompt = f"""
Given the following patient data and guidelines, provide a clinical recommendation.
Do not use a graph. Just output the text recommendation.

Patient Data:
{patient_data}

Guidelines:
{guidelines}
"""
        response = self.api_client.messages.create(
            model=self.model_name,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
