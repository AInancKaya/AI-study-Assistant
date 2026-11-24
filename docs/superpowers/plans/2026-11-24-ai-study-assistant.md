# AI Study Assistant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local, zero-cost app that answers questions about uploaded PDF lecture notes with page citations, generates quizzes and grades answers.

**Architecture:** `src/study_assistant` holds pure-Python core modules (ingest → chunk → embed → Chroma store → RAG / quiz / grading) behind small Protocols (`Embedder`, `LLMClient`) so tests run with fakes. A FastAPI app wires them together; a Streamlit UI talks to the API over HTTP. LLM is Ollama on localhost.

**Tech Stack:** Python 3.12 (uv), FastAPI, Streamlit, Ollama (`ollama` client), sentence-transformers (`intfloat/multilingual-e5-small`), ChromaDB, pypdf, Pydantic v2 / pydantic-settings, SQLite (stdlib), pytest, ruff, pre-commit, GitHub Actions, Docker.

**Spec:** `docs/superpowers/specs/2026-11-24-ai-study-assistant-design.md`

## Global Constraints

- No paid APIs, no API keys at runtime. LLM = Ollama at `http://localhost:11434`, default model `gemma2:9b`.
- Python pinned to 3.12 via `uv` (`.python-version`); dependencies locked in `uv.lock`.
- License MIT; no AGPL dependencies (no PyMuPDF).
- Model names, paths and tuning numbers live only in `config.py` (env prefix `STUDY_`).
- Unit tests never touch the network, Ollama, or the real embedding model. Tests needing the real embedder are marked `@pytest.mark.slow`; tests needing Ollama are marked `@pytest.mark.live`. Default `pytest` run excludes both (`-m "not slow and not live"` in `pyproject.toml` addopts).
- Code, comments, README: English. Learning notes `docs/learning/NN-<topic>.md`: Turkish, each with sections **Ne yaptık? / Neden? / Nasıl çalışıyor? / Önemli kavramlar / Mülakatta sorulabilecek sorular**.
- Context budget: prompts sized for an 8k-token model window.
- Every commit message ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; commit and `git push` at the end of each task.
- Builder model per task (author's request): Opus = skeleton/core, Haiku = simple modules, UI, README.

## Review Focus

1. **Ollama not running or model not pulled** → API answers 503 with the exact command to fix (`ollama serve` / `ollama pull gemma2:9b`); UI shows a banner instead of a stack trace. Tests: Task 6, Task 10.
2. **Scanned / empty PDF, non-PDF file, or >300 pages** → 422 / 400 / 413 with a readable message, nothing half-stored. Tests: Task 2, Task 10.
3. **Question unrelated to the notes** → `found=false`, "not found in your notes" message, zero citations, LLM not called when retrieval is below `min_score`. Tests: Task 7.
4. **LLM returns malformed JSON or bad ids** (source ids out of range, MC with ≠4 options, correct_option out of range) → one retry, then invalid items dropped, never a 500. Tests: Task 6, 7, 8.
5. **Turkish characters (ı, İ, ş, ğ, ç, ö, ü) and line-end hyphenation** survive extraction and chunking. Tests: Task 2, Task 3.

---

## Shared interfaces (defined in Task 1, used everywhere)

`src/study_assistant/models.py`:

```python
class Page(BaseModel):        doc_id: str; filename: str; page: int  # 1-based; text: str
class Chunk(BaseModel):       id: str; doc_id: str; filename: str; page: int; chunk_idx: int; text: str
class RetrievedChunk(BaseModel): chunk: Chunk; score: float          # cosine similarity
class Citation(BaseModel):    doc_id: str; filename: str; page: int
class Answer(BaseModel):      answer: str; found: bool; citations: list[Citation]
class DocumentInfo(BaseModel): doc_id: str; filename: str; pages: int; chunks: int

QuestionType = Literal["multiple_choice", "open_ended"]
class QuizQuestion(BaseModel): id: str; type: QuestionType; question: str
                               options: list[str] | None; correct_option: int | None
                               reference_answer: str; source_pages: list[int]
class Quiz(BaseModel):        id: str; doc_id: str; filename: str; created_at: datetime
                               questions: list[QuizQuestion]
    def public(self) -> "QuizPublic"   # strips correct_option and reference_answer
class QuizQuestionPublic(BaseModel): id; type; question; options
class QuizPublic(BaseModel):  id; doc_id; filename; questions: list[QuizQuestionPublic]

Verdict = Literal["correct", "partially_correct", "incorrect"]
class Grade(BaseModel):       score: int; verdict: Verdict; feedback: str; citations: list[Citation]
                               correct_option: int | None = None; reference_answer: str

# LLM output schemas (what the model must return; no ids, no numeric constraints)
class LLMAnswer(BaseModel):   answer: str; found: bool; cited_sources: list[int]
class LLMQuizQuestion(BaseModel): type: QuestionType; question: str; options: list[str] | None
                               correct_option: int | None; reference_answer: str; source_ids: list[int]
class LLMQuiz(BaseModel):     questions: list[LLMQuizQuestion]
class LLMGrade(BaseModel):    score: int; feedback: str
```

`src/study_assistant/config.py`:

```python
class Settings(BaseSettings):  # env_prefix="STUDY_", env_file=".env"
    ollama_host: str = "http://localhost:11434"
    llm_model: str = "gemma2:9b"
    llm_timeout_s: float = 180.0
    embedding_model: str = "intfloat/multilingual-e5-small"
    data_dir: Path = Path("data")          # chroma at data/chroma, sqlite at data/quizzes.db
    chunk_size: int = 1500                 # characters
    chunk_overlap: int = 200
    top_k: int = 6
    min_score: float = 0.75                # tune with evals (Task 12)
    quiz_max_chunks: int = 10
    max_upload_mb: int = 50
    max_pages: int = 300
    api_url: str = "http://localhost:8000" # used by the UI
def get_settings() -> Settings            # lru_cache'd
```

---

### Task 1: Project scaffold, config and models  *(Opus)*

**Files:**
- Create: `pyproject.toml`, `.python-version`, `uv.lock`, `LICENSE` (MIT, holder "AInancKaya"), `.pre-commit-config.yaml`, `.github/workflows/ci.yml`, `.env.example`, `Makefile`, `src/study_assistant/__init__.py`, `src/study_assistant/config.py`, `src/study_assistant/models.py`, `tests/__init__.py`, `tests/conftest.py`, `tests/test_models.py`, `docs/learning/01-proje-iskeleti.md`

**Interfaces:**
- Produces: everything in "Shared interfaces" above.

- [ ] **Step 1: Install uv and create the project**
Run: `brew install uv && uv python install 3.12 && uv init --lib --name study-assistant --python 3.12 .` (keep existing `.gitignore`, `docs/`). Add deps: `uv add fastapi "uvicorn[standard]" python-multipart pydantic pydantic-settings pypdf chromadb sentence-transformers ollama streamlit httpx`; dev: `uv add --dev pytest pytest-cov ruff pre-commit reportlab`.
Expected: `uv run python -c "import study_assistant"` exits 0.

- [ ] **Step 2: Configure tooling in `pyproject.toml`**
ruff: line-length 100, `select = ["E","F","I","B","UP","SIM"]`. pytest: `testpaths=["tests"]`, markers `slow`, `live`, `addopts = "-m 'not slow and not live' --cov=study_assistant --cov-report=term-missing"`. Pre-commit: `ruff`, `ruff-format`, `uv-lock` hooks. CI (`ci.yml`): on push/PR, `astral-sh/setup-uv`, `uv sync --locked`, `uv run ruff check .`, `uv run ruff format --check .`, `uv run pytest`. A second job `eval` (Task 12 fills it in) may be left out for now. Makefile targets: `setup` (uv sync + `ollama pull $(MODEL)`), `api`, `ui`, `run` (both), `test`, `lint`, `eval`, `live-eval`.

- [ ] **Step 3: Write failing tests `tests/test_models.py`**
```python
def test_quiz_public_hides_answers():
    q = QuizQuestion(id="q1", type="multiple_choice", question="?", options=["a","b","c","d"],
                     correct_option=2, reference_answer="c", source_pages=[3])
    quiz = Quiz(id="z", doc_id="d", filename="f.pdf", created_at=datetime.now(UTC), questions=[q])
    pub = quiz.public().model_dump()
    assert "correct_option" not in pub["questions"][0]
    assert "reference_answer" not in pub["questions"][0]

def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("STUDY_LLM_MODEL", "qwen3:8b")
    assert Settings().llm_model == "qwen3:8b"

def test_llm_schemas_have_no_numeric_constraints():
    for m in (LLMAnswer, LLMQuiz, LLMGrade):
        s = json.dumps(m.model_json_schema())
        assert "minimum" not in s and "maxLength" not in s
```
- [ ] **Step 4:** Run `uv run pytest tests/test_models.py -v` → FAIL (import error).
- [ ] **Step 5:** Implement `config.py` and `models.py` exactly as in Shared interfaces.
- [ ] **Step 6:** Run `uv run pytest -v && uv run ruff check . && uv run ruff format --check .` → all pass.
- [ ] **Step 7:** Write `docs/learning/01-proje-iskeleti.md` (uv, src layout, pydantic, settings, CI, why fakes).
- [ ] **Step 8:** Commit `chore: scaffold project with uv, ruff, pytest, CI` and push.

---

### Task 2: PDF ingestion  *(Haiku)*

**Files:**
- Create: `src/study_assistant/ingest/__init__.py`, `src/study_assistant/ingest/pdf.py`, `scripts/make_pdf.py`, `tests/fixtures/turkce_notlar.pdf`, `tests/fixtures/empty.pdf`, `tests/test_pdf.py`, `docs/learning/02-pdf-okuma.md`

**Interfaces:**
- Consumes: `Page` (Task 1).
- Produces:
  - `class ScannedPDFError(ValueError)`, `class InvalidPDFError(ValueError)`
  - `make_doc_id(data: bytes) -> str` — first 12 hex chars of sha256
  - `extract_pages(data: bytes, filename: str) -> list[Page]` — 1-based pages, skips pages with no text, NFC-normalized, joins line-end hyphenation (`"öğren-\nme"` → `"öğrenme"`), collapses runs of spaces. Raises `InvalidPDFError` if pypdf cannot read it, `ScannedPDFError` if no page has ≥ 20 non-whitespace chars.
  - `count_pages(data: bytes) -> int`
  - `scripts/make_pdf.py`: `make_pdf(pages: list[str], out: Path) -> None` using reportlab with a Unicode TTF font (macOS `/System/Library/Fonts/Supplemental/Arial Unicode.ttf`, fallback `Arial.ttf`), CLI `python scripts/make_pdf.py in.md out.pdf` where pages in the markdown are separated by a line `---page---`.

- [ ] **Step 1:** Generate fixtures with `scripts/make_pdf.py`: `turkce_notlar.pdf` = 3 pages; page 2 contains the exact sentence `Işık ılık süt içinde ağır ağır yayıldı; İstanbul'da öğrenme-\nnin önemi büyüktür.` (hyphen break written so reportlab draws `öğrenme-` at a line end — draw lines explicitly). `empty.pdf` = 2 blank pages. Commit the generated PDFs.
- [ ] **Step 2: Failing tests**
```python
def test_extract_keeps_turkish_and_pages():
    pages = extract_pages(FIX("turkce_notlar.pdf"), "turkce_notlar.pdf")
    assert [p.page for p in pages] == [1, 2, 3]
    assert "Işık ılık süt" in pages[1].text and "İstanbul" in pages[1].text
def test_hyphenation_joined():         assert "öğrenmenin" in pages_of_fixture()[1].text
def test_doc_id_stable():              assert make_doc_id(b"x") == make_doc_id(b"x") and len(make_doc_id(b"x")) == 12
def test_scanned_pdf_raises():         with pytest.raises(ScannedPDFError): extract_pages(FIX("empty.pdf"), "e.pdf")
def test_not_a_pdf_raises():           with pytest.raises(InvalidPDFError): extract_pages(b"hello", "x.pdf")
def test_count_pages():                assert count_pages(FIX("turkce_notlar.pdf")) == 3
```
- [ ] **Step 3:** Run `uv run pytest tests/test_pdf.py -v` → FAIL.
- [ ] **Step 4:** Implement `pdf.py` with `pypdf.PdfReader(io.BytesIO(data))`.
- [ ] **Step 5:** Run → PASS. Ruff clean.
- [ ] **Step 6:** Learning note `02-pdf-okuma.md` (text layer vs scanned, NFC, why pypdf not PyMuPDF/AGPL).
- [ ] **Step 7:** Commit `feat: extract PDF pages with Turkish-safe normalization` and push.

---

### Task 3: Chunking  *(Haiku)*

**Files:**
- Create: `src/study_assistant/ingest/chunking.py`, `tests/test_chunking.py`, `docs/learning/03-parcalama.md`

**Interfaces:**
- Consumes: `Page`, `Chunk`.
- Produces: `chunk_pages(pages: list[Page], size: int = 1500, overlap: int = 200) -> list[Chunk]`
  - Never crosses page boundaries. Splits on paragraph (`\n\n`), then sentence (`(?<=[.!?…])\s+`), then hard character split for a single over-long sentence.
  - Consecutive chunks on the same page share ≥ `overlap` characters taken from whole trailing sentences where possible (hard-split fallback otherwise).
  - `chunk_idx` increases by 1 across the whole document in page order; `id = f"{doc_id}:{page}:{chunk_idx}"`.

- [ ] **Step 1: Failing tests**
```python
def test_short_page_is_one_chunk():   assert len(chunk_pages([page(1, "Kısa metin.")])) == 1
def test_no_chunk_exceeds_size():     assert all(len(c.text) <= 500 for c in chunk_pages(long_pages(), 500, 80))
def test_chunks_never_cross_pages():  # every chunk text is a substring of its own page text
def test_chunk_idx_sequential_and_ids_unique()
def test_overlap_between_neighbours(): # chunk[i] tail (last 80 chars) shares text with chunk[i+1] head
def test_single_huge_sentence_hard_split(): # 3000 chars no punctuation, size 1000 → 3+ chunks, each ≤ 1000
def test_turkish_text_preserved():    # "ığüşöç İĞÜŞÖÇ" appears intact in output
def test_empty_pages_list_returns_empty()
```
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS.
- [ ] **Step 5:** Learning note `03-parcalama.md` (why chunk, size/overlap trade-off, page metadata for citations).
- [ ] **Step 6:** Commit `feat: page-aware text chunking with overlap` and push.

---

### Task 4: Embeddings + test fakes  *(Opus)*

**Files:**
- Create: `src/study_assistant/embeddings.py`, `tests/fakes.py`, `tests/test_embeddings.py`, `docs/learning/04-embedding.md`

**Interfaces:**
- Produces:
  - `class Embedder(Protocol): def embed_passages(self, texts: list[str]) -> list[list[float]]: ...; def embed_query(self, text: str) -> list[float]: ...`
  - `class E5Embedder` — `__init__(self, model_name: str)`; lazy-loads `SentenceTransformer` on first use; prefixes `"passage: "` / `"query: "`; `normalize_embeddings=True`; batch size 32.
  - `tests/fakes.py: class HashEmbedder(dim: int = 256)` — lowercases (Turkish-aware: `İ→i`, `I→ı` before `.lower()`), tokenizes on `\w+`, hashes each token with `hashlib.md5` to a bucket, L2-normalizes. Deterministic; texts sharing words get higher cosine.
  - `tests/fakes.py: class FakeLLM` — `__init__(self, responses: list[BaseModel | Exception])`; `complete_json(system, user, schema)` pops the next response (raises it if it is an Exception), records `(system, user, schema)` in `self.calls`; `check()` returns None.

- [ ] **Step 1: Failing tests**
```python
def test_hash_embedder_similarity():
    e = HashEmbedder()
    a, b, c = e.embed_passages(["fotosentez bitkiler", "bitkiler fotosentez yapar", "borsa endeksi"])
    assert cos(a, b) > cos(a, c)
def test_hash_embedder_normalized():  assert abs(norm(HashEmbedder().embed_query("x y")) - 1) < 1e-6
@pytest.mark.slow
def test_e5_prefixes_and_dim():
    e = E5Embedder("intfloat/multilingual-e5-small")
    q = e.embed_query("Fotosentez nedir?"); p = e.embed_passages(["Fotosentez, bitkilerin ışıkla besin üretmesidir.", "Enflasyon fiyatların artmasıdır."])
    assert len(q) == 384 and cos(q, p[0]) > cos(q, p[1])
```
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run `uv run pytest tests/test_embeddings.py -v -m ""` → PASS (downloads ~470 MB model once).
- [ ] **Step 5:** Learning note `04-embedding.md` (vectors, cosine similarity, e5 prefixes, why local model).
- [ ] **Step 6:** Commit `feat: local multilingual embedder and test fakes` and push.

---

### Task 5: Vector store  *(Opus)*

**Files:**
- Create: `src/study_assistant/store.py`, `tests/test_store.py`, `docs/learning/05-vektor-veritabani.md`

**Interfaces:**
- Consumes: `Embedder`, `Chunk`, `RetrievedChunk`, `DocumentInfo`.
- Produces: `class VectorStore`:
  - `__init__(self, path: Path, embedder: Embedder, collection: str = "chunks")` — `chromadb.PersistentClient(path)`, collection with `metadata={"hnsw:space": "cosine"}`; embeddings always passed explicitly.
  - `add_document(self, chunks: list[Chunk], pages: int) -> DocumentInfo` — metadata per chunk: `doc_id, filename, page, chunk_idx, doc_pages`.
  - `has_document(self, doc_id: str) -> bool`
  - `search(self, query: str, k: int, doc_id: str | None = None) -> list[RetrievedChunk]` — `score = 1 - distance`, sorted desc.
  - `get_chunks(self, doc_id: str, pages: list[int] | None = None) -> list[Chunk]` — sorted by `chunk_idx`.
  - `list_documents(self) -> list[DocumentInfo]` — sorted by filename.
  - `delete_document(self, doc_id: str) -> bool` — False if absent.

- [ ] **Step 1: Failing tests** (use `tmp_path` + `HashEmbedder`)
```python
def test_search_finds_relevant_chunk_first()          # query "fotosentez" → top chunk page == 2
def test_search_filter_by_doc_id()                    # two docs, filter returns only that doc
def test_list_documents_counts_pages_and_chunks()     # DocumentInfo(pages=3, chunks=N)
def test_delete_document()                            # True then has_document False; delete again False
def test_persists_across_instances(tmp_path)          # new VectorStore on same path sees the doc
def test_get_chunks_filters_pages_and_orders()
def test_search_empty_store_returns_empty_list()
```
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS.
- [ ] **Step 5:** Learning note `05-vektor-veritabani.md`.
- [ ] **Step 6:** Commit `feat: Chroma-backed vector store with page metadata` and push.

---

### Task 6: Ollama LLM client  *(Opus)*

**Files:**
- Create: `src/study_assistant/llm.py`, `tests/test_llm.py`, `docs/learning/06-yerel-llm-ollama.md`

**Interfaces:**
- Produces:
  - `class LLMError(RuntimeError)`, `class LLMUnavailableError(LLMError)`, `class ModelNotFoundError(LLMError)`
  - `class LLMClient(Protocol): def complete_json(self, system: str, user: str, schema: type[T]) -> T: ...; def check(self) -> None: ...`
  - `class OllamaClient(host: str, model: str, timeout: float)`:
    - `complete_json` calls `ollama.Client(host, timeout).chat(model=..., messages=[system, user], format=schema.model_json_schema(), options={"temperature": 0, "num_ctx": 8192})`, validates with `schema.model_validate_json`; on `ValidationError`/`JSONDecodeError` retries once with an extra user message "Your previous reply was not valid JSON for the schema. Reply again with JSON only."; second failure → `LLMError`.
    - Connection errors (`httpx.ConnectError`, `ConnectionError`) → `LLMUnavailableError("Ollama is not running at <host>. Start it with: ollama serve")`.
    - `ollama.ResponseError` with status 404 → `ModelNotFoundError("Model '<m>' is not installed. Run: ollama pull <m>")`.
    - `check()` calls `client.show(model)` mapping errors the same way.

- [ ] **Step 1: Failing tests** (monkeypatch `ollama.Client` with a stub class)
```python
def test_returns_validated_model()          # stub returns '{"score": 80, "feedback": "iyi"}' → LLMGrade
def test_retries_once_on_bad_json()         # stub returns "nope" then valid → ok, 2 calls
def test_raises_after_two_bad_replies()     # → LLMError
def test_connection_error_maps_to_unavailable()  # message contains "ollama serve"
def test_404_maps_to_model_not_found()      # message contains "ollama pull gemma2:9b"
def test_sends_schema_and_temperature_zero()
@pytest.mark.live
def test_live_ollama_roundtrip()            # real gemma2:9b returns a valid LLMGrade
```
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS; also `uv run pytest -m live tests/test_llm.py` with Ollama running → PASS.
- [ ] **Step 5:** Learning note `06-yerel-llm-ollama.md` (local inference, structured output via JSON schema, temperature, retries).
- [ ] **Step 6:** Commit `feat: Ollama client with structured JSON output and clear errors` and push.

---

### Task 7: RAG question answering  *(Opus)*

**Files:**
- Create: `src/study_assistant/prompts.py`, `src/study_assistant/rag.py`, `tests/test_rag.py`, `docs/learning/07-rag.md`

**Interfaces:**
- Consumes: `VectorStore.search`, `LLMClient`, `LLMAnswer`, `Answer`, `Citation`.
- Produces:
  - `prompts.py`: `ANSWER_SYSTEM: str`, `QUIZ_SYSTEM: str`, `GRADE_SYSTEM: str`, `format_sources(chunks: list[Chunk]) -> str` — renders `[1] (dosya.pdf, s. 4)\n<text>` blocks.
  - `NOT_FOUND_MESSAGE = "I couldn't find this in your notes. / Bu bilgiyi notlarında bulamadım."`
  - `ask(question: str, store: VectorStore, llm: LLMClient, *, k: int, min_score: float, doc_id: str | None = None) -> Answer`
    - strip question; empty → `ValueError`.
    - keep retrieved chunks with `score >= min_score`; none left → `Answer(answer=NOT_FOUND_MESSAGE, found=False, citations=[])` **without calling the LLM**.
    - `ANSWER_SYSTEM` rules: answer only from sources, answer in the question's language, cite source numbers in `cited_sources`, set `found=false` if sources don't contain the answer.
    - citations = de-duplicated `(doc_id, filename, page)` of `cited_sources` that are in `1..len(sources)`, in first-cited order; out-of-range ids dropped. If `found` is false → `NOT_FOUND_MESSAGE`, no citations. If found but no valid citation → keep answer, `citations=[]`.

- [ ] **Step 1: Failing tests** (`HashEmbedder`, tmp store with the Turkish fixture chunks, `FakeLLM`)
```python
def test_answer_with_valid_citations()          # LLM cites [1,1,2] → 2 unique citations with right pages
def test_out_of_range_source_ids_dropped()      # cites [1, 9] → only source 1
def test_below_min_score_skips_llm()            # min_score=0.99 → found False, llm.calls == []
def test_llm_says_not_found()                   # LLMAnswer(found=False, ...) → NOT_FOUND_MESSAGE, []
def test_doc_filter_passed_to_store()
def test_empty_question_raises()
def test_prompt_contains_numbered_sources()     # llm.calls[0][1] contains "[1] (turkce_notlar.pdf, s. 2)"
```
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS.
- [ ] **Step 5:** Learning note `07-rag.md` (retrieve→augment→generate, hallucination guard, citation validation).
- [ ] **Step 6:** Commit `feat: grounded question answering with validated citations` and push.

---

### Task 8: Quiz generation and storage  *(Opus)*

**Files:**
- Create: `src/study_assistant/quiz_store.py`, `src/study_assistant/quiz.py`, `tests/test_quiz.py`, `docs/learning/08-quiz-uretimi.md`

**Interfaces:**
- Consumes: `VectorStore.get_chunks`, `LLMClient`, `LLMQuiz`, `Quiz`, `QuizQuestion`, `QUIZ_SYSTEM`, `format_sources`.
- Produces:
  - `class QuizRepository(db_path: Path)` — sqlite table `quizzes(id TEXT PRIMARY KEY, doc_id TEXT, created_at TEXT, data TEXT)`; `save(quiz: Quiz) -> None`, `get(quiz_id: str) -> Quiz | None`, `list(doc_id: str | None = None) -> list[Quiz]` (newest first).
  - `class QuizGenerationError(RuntimeError)`
  - `select_quiz_chunks(chunks: list[Chunk], max_chunks: int) -> list[Chunk]` — evenly spaced indices `round(i * (n-1) / (m-1))` for `m = min(max_chunks, n)`, first and last always included, order kept.
  - `generate_quiz(doc_id: str, store: VectorStore, llm: LLMClient, repo: QuizRepository, *, n_multiple_choice: int = 3, n_open_ended: int = 2, max_chunks: int) -> Quiz`
    - counts each 0..10, total 1..10 else `ValueError`; unknown doc → `KeyError`.
    - valid MC: exactly 4 non-empty options, `correct_option in 0..3`; valid open-ended: `options` None/empty, non-empty `reference_answer`. Invalid items dropped; extra items beyond requested counts trimmed per type.
    - `source_pages` from valid `source_ids`; if none valid, all pages of the selected chunks.
    - zero valid questions → `QuizGenerationError`. Ids: `uuid4().hex[:12]` for quiz, `q1..qN` for questions. Saved via `repo.save`.

- [ ] **Step 1: Failing tests**
```python
def test_select_chunks_even_spread()         # 100 chunks, max 10 → 10 chunks, first idx 0, last idx 99
def test_select_chunks_fewer_than_max()      # 3 chunks, max 10 → all 3
def test_generate_valid_quiz_saved()         # FakeLLM returns 3 MC + 2 open → repo.get(quiz.id) == quiz
def test_invalid_questions_dropped()         # MC with 3 options, correct_option 7 → dropped
def test_source_ids_mapped_to_pages()
def test_all_invalid_raises()                # → QuizGenerationError
def test_bad_counts_raise()                  # (0,0) and (8,5) → ValueError
def test_unknown_doc_raises()                # → KeyError
def test_repository_roundtrip_and_list_order(tmp_path)
```
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS.
- [ ] **Step 5:** Learning note `08-quiz-uretimi.md`.
- [ ] **Step 6:** Commit `feat: quiz generation with validation and SQLite storage` and push.

---

### Task 9: Grading  *(Haiku)*

**Files:**
- Create: `src/study_assistant/grading.py`, `tests/test_grading.py`, `docs/learning/09-puanlama.md`

**Interfaces:**
- Consumes: `Quiz`, `VectorStore.get_chunks`, `LLMClient`, `LLMGrade`, `Grade`, `GRADE_SYSTEM`.
- Produces:
  - `verdict_for(score: int) -> Verdict` — ≥80 correct, ≥40 partially_correct, else incorrect.
  - `grade_answer(quiz: Quiz, question_id: str, store: VectorStore, llm: LLMClient, *, answer: str | None = None, selected_option: int | None = None) -> Grade`
    - unknown question → `KeyError`.
    - MC: needs `selected_option` (else `ValueError`); score 100/0, no LLM call; feedback states the correct option text; `correct_option` filled.
    - Open-ended: blank answer → score 0, verdict incorrect, no LLM call. Else LLM gets question, reference answer, source chunks (`get_chunks(doc_id, source_pages)`, max 4) and the student answer; score clamped to 0..100; verdict from `verdict_for` (LLM verdict not trusted).
    - `citations` = source pages of the question; `reference_answer` always returned.

- [ ] **Step 1: Failing tests**
```python
def test_mc_correct_and_wrong_no_llm()
def test_mc_without_selection_raises()
def test_open_ended_uses_llm_and_clamps()      # LLM score 140 → 100, verdict correct
def test_open_blank_answer_scores_zero_no_llm()
def test_verdict_thresholds()                  # 80→correct, 79→partially_correct, 39→incorrect
def test_unknown_question_raises()
```
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS.
- [ ] **Step 5:** Learning note `09-puanlama.md` (LLM-as-judge, why deterministic MC, why we compute verdict).
- [ ] **Step 6:** Commit `feat: answer grading with reference-based feedback` and push.

---

### Task 10: FastAPI backend  *(Opus)*

**Files:**
- Create: `src/study_assistant/api/__init__.py`, `src/study_assistant/api/main.py`, `src/study_assistant/api/schemas.py`, `tests/test_api.py`, `docs/learning/10-fastapi.md`

**Interfaces:**
- Consumes: all core modules.
- Produces:
  - `create_app(settings: Settings | None = None, *, store: VectorStore | None = None, llm: LLMClient | None = None, quizzes: QuizRepository | None = None) -> FastAPI` — builds real deps from settings when not injected. No module-level app object; run with `uvicorn study_assistant.api.main:create_app --factory` (Makefile `api` target, Dockerfile).
  - `schemas.py`: `AskRequest(question: str, doc_id: str | None = None)`, `QuizRequest(doc_id: str, n_multiple_choice: int = 3, n_open_ended: int = 2)`, `GradeRequest(quiz_id: str, question_id: str, answer: str | None = None, selected_option: int | None = None)`, `HealthResponse(status: str, llm_ok: bool, llm_detail: str, model: str)`.
  - Routes:
    | Method & path | Response | Errors |
    |---|---|---|
    | `GET /health` | `HealthResponse` (always 200; `llm_ok` false + detail when Ollama down) | — |
    | `POST /documents` (multipart `file`) | `DocumentInfo` 201; existing doc_id → 200 same info | 400 not `.pdf`/unreadable, 413 > `max_upload_mb` or > `max_pages`, 422 scanned |
    | `GET /documents` | `list[DocumentInfo]` | — |
    | `DELETE /documents/{doc_id}` | 204 | 404 |
    | `POST /ask` | `Answer` | 400 empty question, 404 unknown doc_id |
    | `POST /quiz` | `QuizPublic` 201 | 400 bad counts, 404 unknown doc, 502 `QuizGenerationError` |
    | `POST /grade` | `Grade` | 404 unknown quiz/question, 400 missing selection |
  - Exception handlers: `LLMUnavailableError`/`ModelNotFoundError` → 503 `{"detail": <message>}`; other `LLMError` → 502.
  - Blocking work runs in sync `def` endpoints (FastAPI threadpool).

- [ ] **Step 1: Failing tests** (`TestClient(create_app(settings_tmp, store=VectorStore(tmp, HashEmbedder()), llm=FakeLLM([...]), quizzes=QuizRepository(tmp/"q.db")))`)
```python
def test_health_reports_llm_down()          # FakeLLM.check raises LLMUnavailableError → llm_ok False, 200
def test_upload_list_delete_flow()
def test_reupload_same_pdf_returns_200_same_id()
def test_upload_non_pdf_400()
def test_upload_scanned_422()
def test_upload_too_many_pages_413()        # settings max_pages=2 with 3-page fixture
def test_ask_returns_answer_with_citations()
def test_ask_ollama_down_503_with_hint()    # FakeLLM raises LLMUnavailableError → 503, "ollama serve" in detail
def test_quiz_hides_answers_then_grade()    # /quiz response lacks reference_answer; /grade MC works
def test_grade_unknown_quiz_404()
```
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS; full suite green.
- [ ] **Step 5:** Manual smoke: `make api`, then `curl -F file=@tests/fixtures/turkce_notlar.pdf localhost:8000/documents` and `curl -X POST localhost:8000/ask -H 'content-type: application/json' -d '{"question":"İstanbul hakkında ne deniyor?"}'` → answer with page 2 citation.
- [ ] **Step 6:** Learning note `10-fastapi.md` (REST, dependency injection, status codes, why `create_app`).
- [ ] **Step 7:** Commit `feat: FastAPI backend for documents, ask, quiz and grading` and push.

---

### Task 11: Streamlit UI  *(Haiku)*

**Files:**
- Create: `ui/api_client.py`, `ui/app.py`, `tests/test_api_client.py`, `docs/learning/11-arayuz.md`

**Interfaces:**
- Consumes: the HTTP API of Task 10.
- Produces:
  - `class ApiError(Exception)` (carries `status` and `detail`); `class ApiClient(base_url: str, timeout: float = 300)` with `health()`, `upload(filename: str, data: bytes)`, `documents()`, `delete(doc_id)`, `ask(question, doc_id=None)`, `quiz(doc_id, n_mc, n_open)`, `grade(quiz_id, question_id, answer=None, selected_option=None)` — returns parsed dicts; non-2xx → `ApiError` with server `detail`.
  - `ui/app.py`: page title "AI Study Assistant", sidebar = health status (red banner with `llm_detail` if not ok; error if API unreachable with "Run: make api"), document upload + list with delete. Tabs: **Ask** (document picker with "All notes", chat history in `st.session_state`, citations rendered as `📄 file.pdf · p. 4` chips, spinner while waiting), **Quiz** (pick doc, counts, generate, MC as radio, open as text_area, "Check" per question shows score/verdict/feedback/reference answer/pages).

- [ ] **Step 1: Failing tests** for `ApiClient` with `httpx.MockTransport`: `test_ask_posts_json`, `test_error_detail_raised_as_api_error`, `test_upload_sends_multipart`.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement `api_client.py` (accept an optional `transport` arg for tests) and `app.py`. **Step 4:** Run → PASS.
- [ ] **Step 5:** Manual check: `make run`, upload fixture, ask, generate quiz, grade; take `docs/images/screenshot.png` (browser screenshot) for the README.
- [ ] **Step 6:** Learning note `11-arayuz.md`.
- [ ] **Step 7:** Commit `feat: Streamlit UI for upload, chat and quizzes` and push.

---

### Task 12: Evaluation  *(Opus)*

**Files:**
- Create: `evals/data/sample_notes.md`, `evals/data/sample_notes.pdf`, `evals/data/gold.jsonl`, `evals/retrieval_eval.py`, `evals/live_eval.py`, `evals/results/.gitkeep`, `tests/test_retrieval_eval.py`, `docs/learning/12-degerlendirme.md`
- Modify: `.github/workflows/ci.yml` (add `eval` job), `src/study_assistant/config.py` (only if `min_score` changes)

**Interfaces:**
- Produces:
  - `sample_notes.md`: self-written Turkish lecture notes "Makine Öğrenmesine Giriş", 8 pages (`---page---` separated), distinct topic per page. PDF built with `scripts/make_pdf.py`.
  - `gold.jsonl`: 24 lines `{"question": str, "expected_pages": [int], "lang": "tr"|"en"}` — 18 Turkish, 6 English; plus 4 lines with `"expected_pages": []` (off-topic questions).
  - `retrieval_eval.py`: `evaluate(store, gold, k) -> dict` with `hit_at_k`, `mrr`, `n` over answerable items, and `offtopic_max_score` / suggested `min_score` (midpoint between lowest relevant top-1 score and highest off-topic top-1 score); CLI prints a markdown table and writes `evals/results/retrieval.md`; exits 1 if `hit_at_k < 0.8`.
  - `live_eval.py`: runs `ask` on every gold item with real Ollama; judge prompt (same local model) returns `{"grounded": bool, "correct_pages": bool}`; writes `evals/results/live_eval.md` with grounded %, citation-page accuracy %, not-found accuracy on off-topic items, avg latency.

- [ ] **Step 1: Failing test** `test_evaluate_metrics_math` — fake store returning known rankings → exact `hit_at_k` and `mrr` values (e.g. ranks [1, 2, None] → hit@6 = 2/3, MRR = (1 + 0.5 + 0)/3).
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS.
- [ ] **Step 5:** Run `uv run python evals/retrieval_eval.py` with E5 → hit@6 ≥ 0.8; set `min_score` in config to the suggested value; re-run full tests.
- [ ] **Step 6:** With Ollama running: `uv run python evals/live_eval.py`; commit both result files.
- [ ] **Step 7:** CI `eval` job: cache `~/.cache/huggingface`, run `uv run python evals/retrieval_eval.py`.
- [ ] **Step 8:** Learning note `12-degerlendirme.md` (hit@k, MRR, LLM-as-judge, why evals matter).
- [ ] **Step 9:** Commit `feat: retrieval and groundedness evals with results` and push.

---

### Task 13: Docker and run scripts  *(Haiku)*

**Files:**
- Create: `Dockerfile`, `docker-compose.yml`, `.dockerignore`, `scripts/run_local.sh`, `docs/learning/13-docker.md`

**Interfaces:**
- Produces: multi-stage `Dockerfile` (`ghcr.io/astral-sh/uv:python3.12-bookworm-slim` builder → `python:3.12-slim-bookworm` runtime, non-root user) with one image used by two compose services: `api` (`uvicorn study_assistant.api.main:create_app --factory --host 0.0.0.0 --port 8000`) and `ui` (`streamlit run ui/app.py --server.port 8501`, `STUDY_API_URL=http://api:8000`). Ollama stays on the host: `STUDY_OLLAMA_HOST=http://host.docker.internal:11434`. Volumes: `./data:/app/data`, `hf-cache`. `scripts/run_local.sh`: checks `ollama` reachable (prints fix), checks model pulled, starts API and UI, traps Ctrl-C to stop both.

- [ ] **Step 1:** `docker compose config` → valid. If Docker is installed: `docker compose build` → success (skip and note in report if Docker is absent).
- [ ] **Step 2:** `bash scripts/run_local.sh` → both ports respond (`curl -s localhost:8000/health`).
- [ ] **Step 3:** Learning note `13-docker.md`.
- [ ] **Step 4:** Commit `build: Docker image, compose file and local run script` and push.

---

### Task 14: README and learning index  *(Haiku writes, Opus reviews)*

**Files:**
- Create: `README.md` (replace uv's stub), `docs/learning/README.md`, `docs/architecture.md`

**Interfaces:**
- Produces: `README.md` sections in order: title + one-line pitch + badges (CI, Python 3.12, License MIT); screenshot; **Features**; **Why it's interesting** (local-only, zero cost, grounded citations, evals); **Architecture** (Mermaid flowchart: PDF → pypdf → chunker → e5 → Chroma; question → retriever → Ollama → answer + citations); **Quickstart** (`brew install uv ollama`, `ollama pull gemma2:9b`, `make setup`, `make run`, open `localhost:8501`); **Configuration** table (every `Settings` field + env var); **Model choices** (gemma2:9b default; `qwen3:8b`, `gemma3:4b` for 8 GB machines); **Evaluation** (numbers copied from `evals/results/*.md`); **Testing** (`make test`, markers); **Project structure**; **Roadmap** (React frontend, OCR, more formats, cloud LLM option); **License**. `docs/learning/README.md`: Turkish index linking notes 01–13 in order with one line each. `docs/architecture.md`: module diagram + request sequence for `/ask`.

- [ ] **Step 1:** Write the files. All numbers in README must come from committed result files.
- [ ] **Step 2:** Verify every command in Quickstart by running it (except installs already present); verify all relative links resolve (`grep -o '](\S*)' README.md`, check each path exists).
- [ ] **Step 3:** Opus review pass of README for accuracy against code.
- [ ] **Step 4:** Commit `docs: README, architecture and learning index` and push.

---

## Final verification

- [ ] `uv run ruff check . && uv run ruff format --check . && uv run pytest` → all green.
- [ ] `uv run pytest -m "slow or live"` with Ollama running → green.
- [ ] GitHub Actions run on `main` → green (`gh run list --limit 1`).
- [ ] End-to-end in the browser: upload `evals/data/sample_notes.pdf`, ask one Turkish and one English question (citations correct), ask an off-topic question (not found), generate quiz, grade one MC and one open answer.
