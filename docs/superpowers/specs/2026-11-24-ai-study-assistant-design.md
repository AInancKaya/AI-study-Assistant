# AI Study Assistant — Design

Date: 2026-11-24 · Status: approved (run as one batch, local LLM, no API cost)

## 1. Intent and why

A portfolio project for the author's CV and GitHub. The author wants to show
recruiters a real, working AI application, and **to learn how it works while
building it**: every step is explained (in Turkish) as it is built, not just
generated.

The product: upload PDF lecture notes (Turkish or English), then

1. **Ask** questions and get answers grounded in the notes, with page citations.
2. **Generate a quiz** (multiple choice + open-ended) from a document.
3. **Get graded**: submit an answer to an open-ended question, receive a score
   and feedback that points back to the relevant pages.

It demonstrates the AI skills employers look for: LLM API usage, RAG
(embeddings + vector store), structured output, prompt design, evaluation,
a backend API, a UI, tests and CI.

## 2. Goals, non-goals, anti-goals

**Goals**
- Runs fully locally with one command after setup: **no API keys, no API cost**
  (author decision 2026-11-24). LLM runs in Ollama on the author's machine.
- Answers cite real pages from the uploaded PDF.
- Clean, tested, typed code with CI passing on GitHub.
- A README a recruiter understands in 30 seconds (screenshot/GIF, features,
  architecture diagram, how to run, eval numbers).
- A steady, meaningful commit history: one small commit per plan step.
- A Turkish learning note per step explaining what was built and why.

**Non-goals (for v1)**
- User accounts, auth, multi-user.
- OCR for scanned PDFs (detect and warn instead).
- Cloud deployment (a Dockerfile is in scope; hosting is not).
- Non-PDF formats.

**Anti-goals (failure even if it "works")**
- Answers that invent facts or cite pages that were not retrieved.
- Code the author cannot explain in an interview.
- A repo that looks like one giant AI-generated dump (one huge commit, no tests).
- Tests that call the real API in CI (cost, flakiness, leaked keys).

## 3. Constraints

- macOS Apple Silicon, author's main language is Python.
- Zero running cost: no paid APIs at runtime. Must run on a 16 GB Apple M2
  alongside the embedding model.
- Permissive license (MIT); no AGPL dependencies.
- No secrets needed; config via optional `.env` (already git-ignored).

## 4. Approaches considered

| Approach | Trade-off | Verdict |
|---|---|---|
| **A. Python: FastAPI backend + Streamlit UI** | Fast to build, all-Python so the author can follow every line; Streamlit looks less "custom" than a React app | **Chosen** — learning + speed matter most |
| B. FastAPI + Next.js/React frontend | More impressive UI, shows full-stack skills; doubles the languages and setup to learn | Possible v2 (separate repo later = more GitHub content) |
| C. Streamlit only, no API | Simplest; but no backend/API skills shown and logic tangles with UI | Rejected — weaker on a CV |

Within A, two smaller choices (from the Sonnet research report):
- **Embeddings:** local `intfloat/multilingual-e5-small` (free, offline,
  multilingual) behind an `Embedder` interface.
  Rejected: Voyage AI (paid API, conflicts with the zero-cost goal).
- **Vector store:** ChromaDB persistent client (stores page metadata natively).
  Rejected: FAISS (no metadata), LanceDB (fine, less familiar docs).

## 5. Decided parts of the how

**Tooling:** Python 3.12 pinned via `uv`; `ruff` (lint+format); `pytest` +
`pytest-cov`; `pre-commit`; GitHub Actions CI (ruff + pytest, no API key);
multi-stage `Dockerfile`; MIT license.

**Libraries:** `ollama`, `pypdf`, `sentence-transformers`, `chromadb`,
`fastapi`, `uvicorn`, `pydantic-settings`, `streamlit`, `httpx`.

**LLM:** local via **Ollama** (`ollama` Python client, server at
`http://localhost:11434`). Default model `gemma2:9b` (already installed on the
author's machine, good Turkish, ~5.4 GB); configurable via `LLM_MODEL` env
(README lists alternatives such as `qwen3:8b`, `gemma3:4b` for lower RAM).
Quiz and grading use Ollama structured output (`format=<Pydantic JSON schema>`,
`temperature=0`) and are validated with Pydantic, with one retry on invalid
JSON. Context budget assumes an 8k-token model window. The `LLMClient`
protocol keeps a cloud provider pluggable later, but none ships in v1.
If Ollama is not running or the model is missing, the API returns a clear
error telling the user what command to run.

**Layout**
```
src/study_assistant/
  config.py          # settings from env (.env)
  models.py          # Pydantic models: Chunk, Answer, Citation, Quiz, Grade
  ingest/pdf.py      # PDF -> pages (pypdf, NFC normalize, scanned-PDF check)
  ingest/chunking.py # pages -> chunks (per page, ~1500 chars, ~15% overlap)
  embeddings.py      # Embedder protocol, E5Embedder (local)
  store.py           # VectorStore over Chroma (add, search, list, delete docs)
  llm.py             # LLMClient protocol + Ollama implementation
  rag.py             # ask(question) -> Answer with citations
  quiz.py            # generate_quiz(doc_id, n) -> Quiz
  grading.py         # grade(question, user_answer) -> Grade
  api/main.py        # FastAPI: /documents, /ask, /quiz, /grade, /health
ui/streamlit_app.py  # talks to the API over HTTP
tests/               # unit tests with fake LLM + fake embedder; fixtures PDFs
evals/               # gold Q&A set + retrieval eval script (hit@k, MRR)
docs/learning/       # Turkish step-by-step learning notes
```

**Data flow:** upload PDF → extract pages → chunk (metadata: `doc_id`,
`filename`, `page`, `chunk_idx`) → embed (`passage:` prefix) → Chroma.
Question → embed (`query:` prefix) → top-k (k=6) chunks → Claude with chunks
labeled by page → answer + cited pages. Citations are validated: any page not
among retrieved chunks is dropped. If nothing relevant is retrieved, the
answer says "not found in your notes" instead of guessing.

**Testing:** LLM and embedder are injected interfaces; unit tests use fakes
(no network). Live API tests are marked `@pytest.mark.live` and skipped in CI.
A small Turkish fixture PDF tests extraction of ı/ş/ğ.

**Eval:** 20–30 gold Q&A pairs with expected pages over a sample PDF; script
reports retrieval hit@k and MRR; results go in the README.

**Build process (author's request):**
- Research → Sonnet subagent (done for stack choice).
- Author chose: build in one batch (no pause per step); learning notes still
  written per step.
- Main skeleton (architecture, interfaces, RAG core, API) → Opus.
- Simple code (helpers, small modules, tests boilerplate, UI widgets) → Haiku.
- README → Haiku, reviewed by Opus.
- Each step: small conventional commit, pushed; a short Turkish note in
  `docs/learning/NN-<topic>.md` explaining what/why/how, plus a chat summary.
- Repo docs and code comments are in English (recruiter audience); learning
  notes are in Turkish (for the author).

**Calls made after builder review (Claude's calls, author may override):**
- **Ask scope:** searches all uploaded documents by default, optional
  `doc_id` filter. Citations carry `filename` + `page`. UI has a document
  picker with an "All notes" option.
- **Quiz storage and grading:** quiz generation stores each question with a
  reference answer and source pages (SQLite file next to Chroma, ids per quiz
  and question). `/grade` takes `quiz_id`, `question_id`, `user_answer`;
  grading compares against the stored reference + source chunks, so scores
  are stable and citations match the question. Grading free-form user-written
  questions is out of scope for v1.
- **Quiz source material:** chunks are sampled evenly across the document
  (cap ~10 chunks per quiz, to fit an 8k context), so a 20-page handout and a 300-page course pack
  both work at bounded cost. Supported size target: up to ~300 pages.
- **Evaluation:** two levels. (1) Retrieval hit@k / MRR — offline, runs in
  CI. (2) Answer groundedness — a manual `evals/live_eval.py` run with the
  local Ollama model as judge over the gold set (free, needs Ollama), results
  committed and shown in the README. Never run in CI.

## 6. Left to the builder

- Exact prompts, chunk size tuning, k value tuning within the stated ranges.
- Streamlit page layout details.
- Sample PDF choice for demo/eval (must be freely licensed or self-written).
- Exact CI/Docker file contents.
