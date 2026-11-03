import { useEffect, useState } from "react";

import { ApiError, api } from "../services/api";
import type { EvaluationRun } from "../types/api";

interface Props {
  notify: (kind: "error" | "success" | "info", message: string) => void;
}

const RETRIEVAL_LABELS: Record<string, string> = {
  mrr: "Mean reciprocal rank",
  latency_ms_p50: "Median latency (ms)",
  latency_ms_p95: "95th percentile latency (ms)",
};

const GENERATION_LABELS: Record<string, string> = {
  citation_validity: "Citation validity",
  citation_correctness: "Citation correctness",
  page_accuracy: "Page accuracy",
  groundedness: "Groundedness (lexical)",
  answer_relevance: "Answer relevance (lexical)",
  correct_abstention_rate: "Correct abstention on unanswerable",
  false_abstention_rate: "False abstention on answerable",
  latency_ms_p50: "Median latency (ms)",
};

function fmt(key: string, value: unknown) {
  if (value === null || value === undefined) return "Not available";
  if (typeof value !== "number") return String(value);
  return key.includes("latency") ? value.toFixed(1) : value.toFixed(3);
}

function labelFor(key: string, labels: Record<string, string>) {
  const recall = /^recall_at_(\d+)$/.exec(key);
  return recall ? `Recall@${recall[1]}` : labels[key];
}

function MetricTable({
  title,
  data,
  labels,
}: {
  title: string;
  data?: Record<string, unknown>;
  labels: Record<string, string>;
}) {
  if (!data) return null;
  const rows = Object.keys(data).filter((key) => labelFor(key, labels));
  return (
    <section className="metric-group">
      <h2>{title}</h2>
      <table className="metric-table">
        <tbody>
          {rows.map((key) => (
            <tr key={key}>
              <th scope="row">{labelFor(key, labels)}</th>
              <td>{fmt(key, data[key])}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}

export function EvaluationPage({ notify }: Props) {
  const [run, setRun] = useState<EvaluationRun | null>(null);
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);

  useEffect(() => {
    api
      .listEvaluationRuns()
      .then((runs) => {
        const latest = runs.items.find((r) => r.status === "completed");
        return latest ? api.getEvaluationRun(latest.id) : null;
      })
      .then(setRun)
      .catch((err: unknown) =>
        notify("error", err instanceof ApiError ? err.message : "Evaluation runs could not be loaded."),
      )
      .finally(() => setLoading(false));
  }, [notify]);

  const start = async () => {
    setRunning(true);
    try {
      setRun(await api.runEvaluation(5));
      notify("success", "Evaluation finished.");
    } catch (err) {
      notify("error", err instanceof ApiError ? err.message : "The evaluation could not be run.");
    } finally {
      setRunning(false);
    }
  };

  const gen = run?.metrics.generation as Record<string, unknown> | undefined;
  const tokens = gen?.tokens as { input: number; output: number; estimated: boolean } | undefined;

  return (
    <section className="evaluation" aria-label="Evaluation">
      <header className="chat-head">
        <div>
          <h1>Evaluation</h1>
          <p className="muted small">
            Measures retrieval and answers against the bundled question set in <code>data/evaluation/default.json</code>.
          </p>
        </div>
        <button type="button" className="button button-primary" onClick={start} disabled={running}>
          {running ? "Running…" : "Run evaluation"}
        </button>
      </header>

      <div className="evaluation-body">
        {loading ? (
          <p className="muted">Loading results…</p>
        ) : !run ? (
          <div className="chat-empty">
            <h2>No evaluation has been run yet</h2>
            <p>
              Load the sample documents with <code>python scripts/seed_demo.py</code>, then press Run evaluation. Results
              are measured, never estimated: nothing appears here until a run completes.
            </p>
          </div>
        ) : (
          <>
            <p className="muted small">
              Run on {new Date(run.created_at).toLocaleString()} with {String(run.config.embedding)} embeddings,{" "}
              {String(run.config.retrieval_mode)} retrieval and the {String(run.config.llm_provider)} answer provider.
            </p>
            <div className="metric-groups">
              <MetricTable title="Retrieval" data={run.metrics.retrieval} labels={RETRIEVAL_LABELS} />
              <MetricTable title="Answers" data={gen} labels={GENERATION_LABELS} />
            </div>
            {tokens && (
              <p className="muted small">
                Tokens: {tokens.input} in, {tokens.output} out{tokens.estimated ? " (estimated from text length)" : ""}.
                Cost: {gen?.estimated_cost_usd != null ? `$${String(gen.estimated_cost_usd)}` : "not configured"}.
              </p>
            )}
            <p className="muted small">
              Groundedness and relevance are word-overlap heuristics, useful for spotting regressions but not proof of
              correctness. Review answers by hand as described in <code>docs/evaluation.md</code>.
            </p>
            <h2>Per question</h2>
            <div className="table-scroll">
              <table className="results-table">
                <colgroup>
                  <col className="col-question" />
                  <col className="col-retrieved" />
                  <col className="col-outcome" />
                  <col />
                </colgroup>
                <thead>
                  <tr>
                    <th scope="col">Question</th>
                    <th scope="col">Retrieved</th>
                    <th scope="col">Outcome</th>
                    <th scope="col">Answer</th>
                  </tr>
                </thead>
                <tbody>
                  {run.results.map((r) => {
                    const expected = r.answerable ? "answer" : "abstain";
                    const ok = r.answerable ? !r.abstained : r.abstained;
                    return (
                      <tr key={r.id}>
                        <td className="question-cell">{r.question}</td>
                        <td data-label="Retrieved">
                          {r.answerable ? (
                            <span className="nowrap">
                              {r.recall_at_k ? `Hit, rank ${Math.round(1 / (r.reciprocal_rank || 1))}` : "Miss"}
                            </span>
                          ) : (
                            "No answer expected"
                          )}
                        </td>
                        <td data-label="Outcome" className={ok ? "outcome-ok" : "outcome-bad"}>
                          <span className="nowrap">{r.abstained ? "Abstained" : "Answered"}</span>{" "}
                          <span className="outcome-expected">(expected to {expected})</span>
                        </td>
                        <td data-label="Answer" className="answer-cell">
                          {r.answer}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </>
        )}
      </div>
    </section>
  );
}
