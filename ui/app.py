"""Streamlit front end: upload lecture PDFs, ask grounded questions, take quizzes.

Run from the repo root with ``make ui`` (the API must be running: ``make api``).
"""

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

# ``streamlit run ui/app.py`` puts ``ui/`` on sys.path, not the repo root, so the
# ``ui`` package would not import without this.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st  # noqa: E402

from study_assistant.config import get_settings  # noqa: E402
from ui.api_client import ApiClient, ApiError  # noqa: E402

Choice = tuple[str, str | None]  # (label, doc_id); doc_id None means "All notes"
ALL_NOTES: Choice = ("All notes", None)
VERDICT_LABELS = {
    "correct": "✅ Correct",
    "partially_correct": "🟡 Partially correct",
    "incorrect": "❌ Incorrect",
}


@st.cache_resource
def get_client() -> ApiClient:
    return ApiClient(get_settings().api_url)


def _call[T](action: Callable[[], T]) -> T | None:
    """Run an API call; show its error in the page and return None on failure."""
    try:
        return action()
    except ApiError as exc:
        st.error(exc.detail)
        return None


def _page_chip(citation: dict[str, Any]) -> None:
    st.badge(f"{citation['filename']} · p. {citation['page']}", icon="📄")


def _document_choices(docs: list[dict[str, Any]]) -> list[Choice]:
    return [(doc["filename"], doc["doc_id"]) for doc in docs]


# --- Sidebar -----------------------------------------------------------------


def render_sidebar(client: ApiClient) -> list[dict[str, Any]]:
    """Show API status, upload and the document list. Returns the stored documents."""
    with st.sidebar:
        st.header("Status")
        render_health(client)
        st.header("Your notes")
        render_upload(client)
        return render_documents(client)


def render_health(client: ApiClient) -> None:
    health = _call(client.health)
    if health is None:
        return
    if health["llm_ok"]:
        st.success(f"Model ready: {health['model']}")
    else:
        st.error(health["llm_detail"])


def render_upload(client: ApiClient) -> None:
    uploaded = st.file_uploader("Add a lecture PDF", type=["pdf"], key="uploader")
    if uploaded is None or not st.button("Index PDF", key="index_pdf"):
        return
    with st.spinner("Reading and indexing the PDF..."):
        info = _call(lambda: client.upload(uploaded.name, uploaded.getvalue()))
    if info is not None:
        st.success(f"{uploaded.name}: {info['pages']} pages indexed.")


def render_documents(client: ApiClient) -> list[dict[str, Any]]:
    docs = _call(client.documents) or []
    if not docs:
        st.caption("No notes yet.")
    for doc in docs:
        _render_document_row(client, doc)
    return docs


def _render_document_row(client: ApiClient, doc: dict[str, Any]) -> None:
    name_col, button_col = st.columns([4, 1])
    name_col.markdown(f"**{doc['filename']}**  \n{doc['pages']} pages")
    if button_col.button("Delete", key=f"delete-{doc['doc_id']}"):
        _delete_document(client, doc["doc_id"])


def _delete_document(client: ApiClient, doc_id: str) -> None:
    try:
        client.delete(doc_id)
    except ApiError as exc:
        st.error(exc.detail)
        return
    st.rerun()


# --- Ask tab -----------------------------------------------------------------


def render_ask(client: ApiClient, docs: list[dict[str, Any]]) -> None:
    if not docs:
        st.info("Upload a lecture PDF in the sidebar to start asking questions.")
        return
    choices = [ALL_NOTES, *_document_choices(docs)]
    index = st.selectbox(
        "Search in",
        options=range(len(choices)),
        format_func=lambda i: choices[i][0],
        key="ask_scope",
    )
    question = st.chat_input("Ask about your notes...")
    if question:
        _answer_question(client, question, choices[index][1])
    for message in st.session_state.get("messages", []):
        _render_message(message)


def _answer_question(client: ApiClient, question: str, doc_id: str | None) -> None:
    """Ask the API and append both the question and the answer to the chat history."""
    messages = st.session_state.setdefault("messages", [])
    messages.append({"role": "user", "content": question})
    with st.spinner("Searching your notes..."):
        answer = _call(lambda: client.ask(question, doc_id))
    if answer is not None:
        messages.append({"role": "assistant", **answer})


def _render_message(message: dict[str, Any]) -> None:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
            return
        if message["found"]:
            st.markdown(message["answer"])
        else:
            st.info(message["answer"])
        for citation in message["citations"]:
            _page_chip(citation)


# --- Quiz tab ----------------------------------------------------------------


def render_quiz(client: ApiClient, docs: list[dict[str, Any]]) -> None:
    if not docs:
        st.info("Upload a lecture PDF in the sidebar to build a quiz.")
        return
    index = st.selectbox(
        "Quiz from",
        options=range(len(docs)),
        format_func=lambda i: docs[i]["filename"],
        key="quiz_doc",
    )
    col_mc, col_open = st.columns(2)
    n_mc = col_mc.slider("Multiple choice", 0, 5, 3, key="quiz_mc")
    n_open = col_open.slider("Open-ended", 0, 5, 2, key="quiz_open")
    if st.button("Generate quiz", type="primary", key="generate_quiz"):
        _generate_quiz(client, docs[index]["doc_id"], n_mc, n_open)
    quiz = st.session_state.get("quiz")
    if quiz is not None:
        _render_quiz(client, quiz)


def _generate_quiz(client: ApiClient, doc_id: str, n_mc: int, n_open: int) -> None:
    with st.spinner("Writing questions with the local model. This can take a minute..."):
        quiz = _call(lambda: client.quiz(doc_id, n_mc, n_open))
    if quiz is not None:
        st.session_state["quiz"] = quiz
        st.session_state["grades"] = {}


def _render_quiz(client: ApiClient, quiz: dict[str, Any]) -> None:
    st.subheader(f"Quiz on {quiz['filename']}")
    for number, question in enumerate(quiz["questions"], start=1):
        with st.container(border=True):
            st.markdown(f"**{number}. {question['question']}**")
            _render_question(client, quiz["id"], question)


def _render_question(client: ApiClient, quiz_id: str, question: dict[str, Any]) -> None:
    key = f"{quiz_id}-{question['id']}"
    if question["type"] == "multiple_choice":
        options = question["options"]
        answer = st.radio(
            "Your answer",
            options=range(len(options)),
            format_func=lambda i: options[i],
            index=None,
            key=f"mc-{key}",
            label_visibility="collapsed",
        )
        check = st.button("Check", key=f"check-{key}")
        if check:
            _check(client, quiz_id, question["id"], selected_option=answer)
    else:
        answer = st.text_area("Your answer", key=f"open-{key}", label_visibility="collapsed")
        check = st.button("Check", key=f"check-{key}")
        if check:
            _check(client, quiz_id, question["id"], answer=answer)
    grade = st.session_state.get("grades", {}).get(question["id"])
    if grade is not None:
        _render_grade(grade, question)


def _check(
    client: ApiClient,
    quiz_id: str,
    question_id: str,
    answer: str | None = None,
    selected_option: int | None = None,
) -> None:
    """Grade one answer and remember the result for this quiz."""
    if answer is not None and not answer.strip():
        st.warning("Write an answer first.")
        return
    if answer is None and selected_option is None:
        st.warning("Pick an option first.")
        return
    grade = _call(
        lambda: client.grade(quiz_id, question_id, answer=answer, selected_option=selected_option)
    )
    if grade is not None:
        st.session_state.setdefault("grades", {})[question_id] = grade


def _render_grade(grade: dict[str, Any], question: dict[str, Any]) -> None:
    st.progress(grade["score"] / 100, text=f"Score: {grade['score']}/100")
    st.markdown(f"**{VERDICT_LABELS.get(grade['verdict'], grade['verdict'])}**")
    st.write(grade["feedback"])
    st.markdown(f"**Reference answer:** {_reference_text(grade, question)}")
    if grade["citations"]:
        st.caption("Where it is in your notes:")
        for citation in grade["citations"]:
            _page_chip(citation)


def _reference_text(grade: dict[str, Any], question: dict[str, Any]) -> str:
    correct = grade["correct_option"]
    if question["type"] == "multiple_choice" and correct is not None:
        return question["options"][correct]
    return grade["reference_answer"]


# --- Page --------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="AI Study Assistant", page_icon="📚", layout="wide")
    st.title("AI Study Assistant")
    st.caption("Ask your lecture notes anything, or test yourself. Runs fully on your machine.")
    client = get_client()
    docs = render_sidebar(client)
    ask_tab, quiz_tab = st.tabs(["Ask", "Quiz"])
    with ask_tab:
        render_ask(client, docs)
    with quiz_tab:
        render_quiz(client, docs)


if __name__ == "__main__":
    main()
