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

Set your API key for Google Gemini:
```bash
export OPENAI_API_KEY="your-gemini-api-key"
```

## Running the Evaluation
To evaluate the architecture against all available datasets in a single, self-healing pass:
```bash
python run_eval.py --bulk-model gemini-2.5-flash --synth-model gemini-3.5-flash
```
- **Self-Healing Server**: `run_eval.py` automatically checks for and starts the local FHIR server using `docker compose up -d` if it's not already running.
- **Transparent Caching**: The pipeline natively supports MIMIC-IV NDJSON format and will automatically invoke and cache the conversion step on first load. No manual conversion needed.
- **Evaluation Arms**: The harness evaluates the case set across 5 distinct architectures (`single-agent+RAG`, `free-text multi-agent`, `VERDICT-verifier-ablated`, `VERDICT-fixed-scheduling`, and `full VERDICT`).
- **Resumable Execution**: The harness automatically streams output to `logs/eval_results_<arm>.jsonl`. If execution is interrupted (e.g., due to power loss), running the script again will skip already-completed patient evaluations. Use `--force-rerun` to restart from scratch.
- **Rate Limiting**: Built-in token bucket rate limiting ensures that free-tier API caps (TPM/RPM) are respected without arbitrary sleeps or manual intervention.

## Evaluation Results
The results of previous runs, ablation studies, and baseline comparisons can be found in `RESULTS.md`.
