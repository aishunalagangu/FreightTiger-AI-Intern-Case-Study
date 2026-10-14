# FreightTiger — Smart Shipping Cost Monitor

## What this solution does
This project processes the supplied 2,940 shipment records into weekly route-level cost metrics, compares each route with its own recent history and peer routes of the same `route_type`, and emits a CSV of anomaly candidates in the required column order. It also checks the supplied context notes before assigning a justified verdict.

## Run it
Requirements: Python 3.10+ and `pandas`.

```bash
python -m pip install pandas
python main.py
```

Files:
- `data/shipment_records.csv` — original shipment input.
- `data/context_notes.csv` — original context notes.
- `data/sample_output_format_v2.csv` — supplied example/column contract.
- `main.py` — deterministic processing pipeline.
- `output.csv` — generated anomaly candidates.
- `evaluation.json` — validation checks.
- `token_cost_log.json` — model/token/cost accounting.
- `reproducibility_check.txt` — result of three untouched runs.

## Calculation choices
1. `week_of` is the Monday of the shipment's Monday–Sunday week.
2. Route key is `origin-destination`; `route_type` remains the provided value.
3. Weekly cost per tonne-km = sum of freight cost / sum of (`quantity_tonnes * distance_km`) for shipments on that route in that week. This correctly weights shipments by tonne-km.
4. Own-history baseline is the arithmetic mean of up to the previous eight *observed route-weeks*, strictly before the current week. No future weeks are used. When fewer than eight are available, all available prior weeks are used; when none exist, the comparison is `N/A`.
5. Peer baseline is the average current-week rate of all other routes with the same `route_type`, excluding the current route. The peer percentage is current rate / peer average - 1.
6. A candidate is emitted when its rate is at least 15% above either available baseline. The 15% threshold is an explicit heuristic because the brief requires anomaly flags but does not prescribe a numeric threshold. It is configurable at the top of `main.py`.
7. `flagged=Yes` means an unexplained candidate requiring human review. `No (justified)` means a candidate matched a positive note that applies to that route and time. A note ID is populated only for justified cases.

## Note grounding / hallucination guardrail
The note matcher is intentionally deterministic and conservative. It checks route scope and a bounded date window where the note describes one. It recognizes the supplied nationwide diesel note only in its stated start week (the note does not define duration), the Chennai–Bangalore flooding interval as Feb 24–Mar 8, 2025, and a festival surcharge only in the week containing the note date. Notes explicitly saying there was no meaningful impact (including N005/N006/N008/N010) cannot justify a rise. A note for a different route or a different period is not accepted. If no positive matching note is found, the candidate remains unexplained.

This is a lightweight retrieval implementation over the provided notes, not a vector database or generative RAG system. That trade-off avoids hallucinated explanations and makes every verdict reproducible. A future enhancement could use embeddings to retrieve candidate notes, while retaining the same strict route/date validation before a note can justify a verdict.

## Evaluation and reproducibility
`evaluation.json` records row counts and checks for required columns, positive rates, unique route-weeks, and note-ID consistency. The provided script is deterministic and uses no random operations. `reproducibility_check.txt` records SHA-256 hashes from three independent executions on the same input. Matching hashes demonstrate identical CSV output.

## Token and cost log
`token_cost_log.json` reports 0 input tokens, 0 output tokens, 0 LLM calls, and $0 model cost because this implementation does not call an LLM. This is accurate rather than an invented estimate. Python execution costs are not model-token costs.

## Known limitations / trade-offs
- The brief does not define the anomaly threshold; 15% is a transparent configurable heuristic and should be discussed in the walkthrough.
- There are few route groups in the sample, so some same-type peer comparisons are based on a small number of peers.
- Note matching uses exact route names and conservative rule-based date windows. It intentionally prefers “unexplained” over unsupported claims.
- Only anomaly candidates are included in `output.csv`; all route-week calculations are available internally in `weekly` during execution. The output contract is preserved exactly.
