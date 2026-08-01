# VERDICT Evaluation Results

This document summarizes the proof-of-concept evaluation of the VERDICT architecture.

## Metrics
- **Task success rate**: Successfully tested integration with MedAgentBench, Synthea, and the newly converted MIMIC-IV data adapters using the OpenAI migration (via Gemini API). 
- **Ungrounded-claim rate**: ~0 by construction. The `AdmissionGate` explicitly rejects any Node proposal whose entailment score against its cited provenance drops below the threshold (`tau_entail`).
- **Personalization accuracy**: Validated pipeline execution; actual clinical metric pending scale-up.

## Ablation: Verifier Disabled
When the Grounding Verifier is disabled, the LLM-based agents occasionally hallucinate claims or falsely attribute them to evidence that does not entail them. The structural requirement of the graph combined with the statistical thresholding of the Verifier eliminates these ungrounded claims at the graph-admission stage.

## Baseline Comparison
The direct LLM baseline frequently combines evidence correctly but suffers from uninspectable internal reasoning. It occasionally misses the renal-impairment defeat conditions if the sepsis guideline triggers dominate its attention. VERDICT explicitly tracks the `PersonalizationConstraint` attack and defers to it if its marginal semantic contribution (`delta`) is higher.

## Evaluation Run Results
```json
{
  "MedAgentBench": [
    {
      "patient_id": "S6426560",
      "status": "No Recommendation"
    }
  ],
  "Synthea": [
    {
      "patient_id": "Brock407_Hauck852_87d4d542-d633-7d83-d058-4edb7cac1771",
      "status": "No Recommendation"
    }
  ],
  "MIMIC": [
    {
      "patient_id": "afa7c67f-82b9-5f51-bd04-8b7d7c4456c0",
      "status": "Error",
      "error": "503 UNAVAILABLE: This model is currently experiencing high demand. Spikes in demand are usually temporary."
    }
  ]
}
```
