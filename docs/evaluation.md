# Evaluation

DocuMind's evaluation separates two questions: **did retrieval find the right passages**, and **did the answer use them correctly**. A good answer can't come from bad retrieval, but good retrieval doesn't guarantee a good answer, so they are measured independently.

## Dataset

`data/evaluation/default.json` (CC0) covers the three bundled sample documents.

```json
{
  "id": "rw-04",
  "question": "How much is the home office stipend for remote employees?",
  "document": "northwind_remote_work_policy.pdf",
  "pages": [3],
  "evidence": ["600 euros"],
  "reference_answer": "Each remote employee receives a one-time home office stipend of 600 euros.",
  "answerable": true
}
```

- `document`: the original filename of the source document.
- `evidence`: phrases a passage must **all** contain to count as relevant (case- and whitespace-insensitive).
- `pages`: the page(s) where the answer appears (empty for text files).
- `answerable: false` marks questions the system should decline: out-of-scope questions and a prompt-injection attempt.

To add a dataset, create `data/evaluation/<name>.json` with the same structure, upload its documents, and run `python scripts/evaluate_rag.py --dataset <name>`.

## Running

```bash
python scripts/seed_demo.py                       # index the sample documents
python scripts/evaluate_rag.py                    # retrieval + answers; stores the run in the database
python scripts/evaluate_rag.py --retrieval-only   # no LLM calls
python scripts/evaluate_rag.py --no-store --top-k 3
```

Output: a JSON file in `data/evaluation/results/` (`<dataset>_<UTC timestamp>.json` and `<dataset>_latest.json`) containing the configuration, aggregate metrics and per-question rows (retrieved passages with relevance flags, answer, citations, latency, usage). Runs stored in the database appear on the Evaluation page.

Retrieval and answering search **all** indexed documents, not only the expected one, so retrieval has to pick the right document as well as the right passage.

## Metrics

### Retrieval (answerable questions only)

| Metric | Definition |
|---|---|
| Recall@K | 1 if any of the top-K passages is relevant, else 0; averaged. (With one relevant passage per question this equals hit rate.) |
| MRR | Mean of `1 / rank` of the first relevant passage (0 if none in the top K). |
| Latency | p50 and p95 of retrieval time per question. |

### Answers

| Metric | Over | Definition |
|---|---|---|
| False abstention rate | answerable | Share answered with `insufficient_evidence` or `refused`. |
| Correct abstention rate | unanswerable | Share correctly declined. |
| Citation validity | answered | Share of citations whose chunk exists and whose excerpt occurs verbatim in it. |
| Citation correctness | answered | Share of citations pointing to a relevant passage of the expected document. Strict: a citation to another passage that also supports the answer counts as incorrect. |
| Page accuracy | answered PDFs | Share where at least one valid citation is on an expected page. |
| Groundedness (lexical) | answered | Share of the answer's content words that appear in its cited passages. A rough faithfulness proxy: it catches answers that drift from their sources, but word overlap doesn't prove a claim is supported. |
| Answer relevance (lexical) | answered | Share of the reference answer's content words present in the answer. |
| Latency | all | p50 and p95 of end-to-end answer time. |
| Tokens and cost | all | Provider-reported usage when available, otherwise estimated (~4 characters per token) and flagged `estimated`. Cost appears only if `DOCUMIND_LLM_COST_PER_1K_INPUT` and `_OUTPUT` are set. |

No LLM-as-judge is used. Model judges are useful but are themselves fallible models; if one is added, treat its scores as another signal to review, not as ground truth.

## Manual review procedure

Automated metrics catch regressions; they don't establish correctness. Review a sample of answers by hand after any change to prompts, providers, chunking or retrieval:

1. Open the latest results JSON (or the Evaluation page) and take every answered question, or at least 20 at random for larger datasets.
2. For each answer, open each citation (`chunk_id`) in the UI evidence panel or via `GET /api/v1/chunks/{id}`.
3. Score each answer on the rubric below and record it in the report template.
4. For each abstention on an answerable question, check whether the evidence was retrieved (`retrieved[].relevant`). If it was, the problem is in generation or the evidence gate; if not, in retrieval.
5. For unanswerable questions that were answered, record what the answer claimed and which passage it cited.

| Score | Faithfulness | Citation |
|---|---|---|
| 2 | Every claim is stated in the cited passages. | Every citation supports the sentence it's attached to. |
| 1 | Mostly supported; minor unsupported detail. | Citations are relevant but some don't support their sentence. |
| 0 | A central claim is unsupported or contradicted. | Citations are missing, wrong or misleading. |

Also mark whether the answer is **complete** (yes / partly / no) and whether anything in `inference` should have been presented as evidence or vice versa.

## Report template

Copy into a new file per evaluation (for example `docs/reports/2026-10-12.md`).

```markdown
# Evaluation report — <date>

## Configuration
- Commit:
- Dataset: default (<n> questions: <a> answerable, <u> unanswerable)
- Embeddings: <provider / model>
- Retrieval: <hybrid|semantic|keyword>, top-k <k>, reranker <name>
- LLM: <provider / model>, temperature <t>
- Chunking: size <s>, overlap <o>

## Retrieval
| Recall@k | MRR | p50 ms | p95 ms |
|---|---|---|---|
|  |  |  |  |

## Answers (automated)
| False abstention | Correct abstention | Citation validity | Citation correctness | Page accuracy | Groundedness | Relevance |
|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |

Tokens: <in> / <out> (estimated: yes/no). Cost: <$ or n/a>.

## Manual review (<n> answers reviewed)
| ID | Faithfulness (0–2) | Citation (0–2) | Complete | Notes |
|---|---|---|---|---|
|  |  |  |  |  |

Mean faithfulness: <x>. Mean citation score: <y>.

## Failure analysis
- Retrieval misses:
- False abstentions:
- Unsupported claims:
- Prompt-injection or refusal issues:

## Changes since last report and next steps
```

## Current results

Two configurations have been measured on the bundled dataset. Results, the comparison and caveats are in the README's [Evaluation section](../README.md#evaluation).

| Configuration | Date | Raw output |
|---|---|---|
| Demo: hashing embedder + extractive mock provider | 2026-10-12 | [`evaluation-results-demo.json`](evaluation-results-demo.json) |
| Real models: `all-MiniLM-L6-v2` + `llama3.2:3b` via Ollama, CPU only | 2026-11-02 | [`evaluation-results-ollama.json`](evaluation-results-ollama.json) |

The real-model run is the third run with that setup. The first two exposed output-format problems in the small model (missing citation labels, abstentions written as prose), which were fixed in the prompt and parser before the reported run. Because those fixes were made while looking at this same question set, the reported numbers are optimistic. No manual review has been recorded yet.
