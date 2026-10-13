import { useState } from "react";

import { ApiError, api } from "../services/api";
import type { Citation, DocumentInfo, QuizQuestion, StudyResponse, StudyTask } from "../types/api";

interface Props {
  documents: DocumentInfo[];
  onOpenCitation: (citation: Citation) => void;
}

const TASKS: { id: StudyTask; label: string; hint: string }[] = [
  { id: "summary", label: "Summary", hint: "A short overview of the document." },
  { id: "key_points", label: "Key points", hint: "The statements that matter most." },
  { id: "revision_notes", label: "Revision notes", hint: "Topic-by-topic notes for review." },
  { id: "quiz", label: "Quiz", hint: "Multiple-choice questions with explained answers." },
];

function SourceLinks({ citations, onOpen }: { citations: Citation[]; onOpen: (c: Citation) => void }) {
  return (
    <span className="item-sources">
      {citations.map((c) => (
        <button key={c.chunk_id} type="button" className="cite-mark" onClick={() => onOpen(c)}>
          {c.page_number ? `p. ${c.page_number}` : c.label}
        </button>
      ))}
    </span>
  );
}

function Quiz({ questions, onOpen }: { questions: QuizQuestion[]; onOpen: (c: Citation) => void }) {
  const [answers, setAnswers] = useState<Record<number, number>>({});
  const answered = Object.keys(answers).length;
  const correct = questions.filter((q, i) => answers[i] === q.correct_index).length;
  return (
    <div className="quiz">
      <p className="muted small" aria-live="polite">
        {answered === questions.length
          ? `You got ${correct} of ${questions.length} right.`
          : `${answered} of ${questions.length} answered.`}
      </p>
      <ol className="quiz-list">
        {questions.map((q, qi) => {
          const chosen = answers[qi];
          const done = chosen !== undefined;
          return (
            <li key={qi} className="quiz-question">
              <fieldset>
                <legend>{q.question}</legend>
                {q.options.map((opt, oi) => {
                  const state = !done ? "" : oi === q.correct_index ? " is-correct" : oi === chosen ? " is-wrong" : "";
                  return (
                    <label key={oi} className={`quiz-option${state}`}>
                      <input
                        type="radio"
                        name={`q-${qi}`}
                        checked={chosen === oi}
                        disabled={done}
                        onChange={() => setAnswers((a) => ({ ...a, [qi]: oi }))}
                      />
                      {opt}
                    </label>
                  );
                })}
              </fieldset>
              {done && (
                <p className="quiz-explanation">
                  {chosen === q.correct_index ? "Correct. " : `The answer is “${q.options[q.correct_index]}”. `}
                  {q.explanation} <SourceLinks citations={q.citations} onOpen={onOpen} />
                </p>
              )}
            </li>
          );
        })}
      </ol>
    </div>
  );
}

export function StudyPage({ documents, onOpenCitation }: Props) {
  const ready = documents.filter((d) => d.status === "completed");
  const [chosenId, setDocId] = useState<string>("");
  const [task, setTask] = useState<StudyTask>("summary");
  const [result, setResult] = useState<StudyResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fall back to the first ready document if the chosen one is gone or not chosen yet.
  const docId = ready.some((d) => d.id === chosenId) ? chosenId : (ready[0]?.id ?? "");

  const generate = async () => {
    if (!docId) return;
    setLoading(true);
    setError(null);
    try {
      setResult(await api.study(docId, task));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Study material could not be generated.");
    } finally {
      setLoading(false);
    }
  };

  if (!ready.length) {
    return (
      <section className="study">
        <div className="chat-empty">
          <h2>Study tools</h2>
          <p>Upload a document and wait until it shows Ready. Then you can generate summaries, notes and quizzes.</p>
        </div>
      </section>
    );
  }

  const title = ready.find((d) => d.id === result?.document_id)?.title;

  return (
    <section className="study" aria-label="Study tools">
      <header className="chat-head">
        <div>
          <h1>Study tools</h1>
          <p className="muted small">Everything generated here links back to the passages it was built from.</p>
        </div>
      </header>
      <div className="study-body">
        <div className="study-controls">
          <label className="field">
            <span>Document</span>
            <select value={docId} onChange={(e) => setDocId(e.target.value)}>
              {ready.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.title}
                </option>
              ))}
            </select>
          </label>
          <fieldset className="task-picker">
            <legend>What to create</legend>
            {TASKS.map((t) => (
              <label key={t.id} className={`task-option${task === t.id ? " is-selected" : ""}`}>
                <input type="radio" name="task" value={t.id} checked={task === t.id} onChange={() => setTask(t.id)} />
                <span className="task-label">{t.label}</span>
                <span className="muted small">{t.hint}</span>
              </label>
            ))}
          </fieldset>
          <button type="button" className="button button-primary" onClick={generate} disabled={loading || !docId}>
            {loading ? "Creating…" : `Create ${TASKS.find((t) => t.id === task)?.label.toLowerCase()}`}
          </button>
          {error && (
            <p className="form-error" role="alert">
              {error}
            </p>
          )}
        </div>

        <div className="study-result" aria-live="polite">
          {loading && (
            <p className="thinking">
              <span className="spinner" aria-hidden="true" /> Reading the document…
            </p>
          )}
          {!loading && result && (
            <>
              <h2>
                {TASKS.find((t) => t.id === result.task)?.label} of {title}
              </h2>
              <p className="muted small">
                Built from {result.passages_used} passages.
                {result.dropped_items > 0 &&
                  ` ${result.dropped_items} generated items were discarded because they cited no valid source.`}
              </p>
              {result.task === "quiz" ? (
                result.questions.length ? (
                  <Quiz key={result.document_id + result.questions.length} questions={result.questions} onOpen={onOpenCitation} />
                ) : (
                  <p>No quiz questions could be grounded in this document's passages.</p>
                )
              ) : result.items.length ? (
                <ul className="study-items">
                  {result.items.map((item, i) => (
                    <li key={i}>
                      {item.text} <SourceLinks citations={item.citations} onOpen={onOpenCitation} />
                    </li>
                  ))}
                </ul>
              ) : (
                <p>Nothing could be generated with valid sources for this document.</p>
              )}
            </>
          )}
          {!loading && !result && <p className="muted">Choose a document and what to create, then press Create.</p>}
        </div>
      </div>
    </section>
  );
}
