# VERDICT: Verifiable Evidence-grounded Reproducible Defeasible Inference for Clinical Trust

**VERDICT** is a research proof-of-concept for a clinical decision-support architecture. Instead of debating in free-text space, LLM agents act as typed operators that propose provenance-gated nodes/edges into a shared **Clinical Argument Graph (CAG)**. The final clinical recommendation, its explanation, its trust score, and its personalization justification are all derived from a single deterministic quantitative bipolar argumentation framework (QBAF) computed over that graph.

## Key Features
- **Typed Argument Graph**: Agents can only submit structured arguments (Evidence, Knowledge, Guidelines, Hypotheses) with explicit provenance.
- **Entailment Grounding Verification**: Node claims are robustly checked against their cited evidence via LLM/NLI entailment thresholds to prevent hallucination.
- **Defeasible Semantics**: Resolves conflicting clinical conditions (e.g., general guidelines vs. specific patient impairments) through argumentation defeat mechanisms.
- **Multi-Dataset Support**: Built-in support for MedAgentBench, Synthea FHIR bundles, and MIMIC-IV FHIR data.

## Project Structure
- `verdict/`: Core package containing the QBAF semantics engine, graph models, orchestrator loop, and specialized agents.
- `scripts/`: Data fetching and preprocessing scripts (e.g., MIMIC NDJSON to bundle conversion).
- `data/`: Directory for patient datasets (Synthea, MIMIC, etc.). *Ignored in git by default.*
- `run_eval.py`: The main evaluation harness connecting the orchestrator, agents, and datasets.

## Setup
Install dependencies:
```bash
pip install -r requirements.txt
```

Set your API key (OpenAI or Gemini):
```bash
export OPENAI_API_KEY="your-api-key"
```

## Running the Evaluation
To evaluate the architecture against available datasets:
```bash
python run_eval.py --model gemini-3.5-flash --limit 1 --sleep 60
```
- `--model`: Specific model to query (defaults to `gemini-3.5-flash`).
- `--limit`: Maximum number of patients to evaluate per dataset to save time/tokens.
- `--sleep`: Cooldown timer between evaluations (useful for free-tier rate limits).

## Evaluation Results
The results of previous runs, ablation studies, and baseline comparisons can be found in `RESULTS.md`.
