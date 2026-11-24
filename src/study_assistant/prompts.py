"""System prompts for the local LLM and the renderer for numbered source blocks.

Every prompt is written for a small local model with an 8k-token window: short, explicit
rules, and the JSON shape is enforced separately by the schema passed to ``complete_json``.
"""

from __future__ import annotations

import re

from study_assistant.models import Chunk

TURKISH_CHARS = frozenset("çğıöşüÇĞİÖŞÜ")
TURKISH_WORDS = frozenset(
    {
        "ve",
        "bir",
        "bu",
        "için",
        "ile",
        "değil",
        "olarak",
        "çünkü",
        "daha",
        "gibi",
        "veya",
        "olan",
        "nedir",
        # ASCII-folded spellings typed without Turkish keyboard support.
        "icin",
        "degil",
        "cunku",
    }
)
ENGLISH_WORDS = frozenset({"the", "and", "is", "of", "to", "because", "it", "a"})
_WORD = re.compile(r"\w+")

ANSWER_SYSTEM = """You are a study assistant. You answer a student's question using ONLY \
the numbered sources from the student's lecture notes given in the user message.

Rules:
- Use only facts stated in the sources. Never use outside knowledge, never guess.
- Answer in the same language as the question (for example Turkish question -> Turkish answer).
- Be clear and concise: a few sentences, or a short list if that is clearer.
- Put the numbers of the sources you used in "cited_sources" (for example [1, 3]). \
Only use numbers that appear in the sources. Do not write the numbers inside "answer".
- If the sources only partly answer the question, answer the part they support and \
say what is missing from the notes.
- If the sources do not contain the answer, set "found" to false, \
"cited_sources" to [] and "answer" to "".
- Otherwise set "found" to true.

Reply with JSON only: {"answer": string, "found": boolean, "cited_sources": [integer]}"""

QUIZ_SYSTEM = """You are a teacher writing exam questions for a student. Use ONLY the \
numbered sources from the student's lecture notes given in the user message.

Rules:
- Every question must be answerable from the sources alone. Never use outside knowledge.
- Ask about the important concepts, definitions, causes and relationships in the content.
- Never ask trivia about page numbers, file names, source numbers or document layout.
- Write each question, its options and its answer in the same language as the sources.
- For "multiple_choice": "options" has exactly 4 plain option texts without prefixes \
such as "A)" or "1.", exactly one option is correct, and "correct_option" is the 0-based \
index of the correct option (0, 1, 2 or 3). The wrong options must be plausible. \
"reference_answer" briefly explains why the correct option is right.
- For "open_ended": "options" is null, "correct_option" is null, and \
"reference_answer" is a concise model answer (1-3 sentences) based on the sources.
- "source_ids" lists the numbers of the sources each question is based on (for example [2]).
- Do not repeat the same question.

Reply with JSON only: {"questions": [{"type": "multiple_choice" or "open_ended", \
"question": string, "options": [string] or null, "correct_option": integer or null, \
"reference_answer": string, "source_ids": [integer]}]}"""

GRADE_SYSTEM = """You are a fair teacher grading a student's answer to an exam question. \
The user message contains the question, the reference answer, the numbered sources from \
the lecture notes, and the student's answer.

Rules:
- Judge the student's answer against the reference answer and the sources only.
- "score" is an integer from 0 to 100: 100 = fully correct and complete, \
0 = wrong or empty. Give partial credit for answers that are partly correct.
- Judge the meaning, not the wording. Ignore spelling and grammar mistakes.
- "feedback" is 1-3 sentences, written in the language named on the last line of the \
user message: say what was right and what was missing or wrong.

Reply with JSON only: {"score": integer, "feedback": string}"""


def format_sources(chunks: list[Chunk]) -> str:
    """Render chunks as numbered source blocks (from 1, in the given order).

    Each block is ``[n] (filename, s. page)`` followed by the chunk text; blocks are
    separated by a blank line. ``s.`` is Turkish "sayfa" (page).
    """
    return "\n\n".join(
        f"[{n}] ({chunk.filename}, s. {chunk.page})\n{chunk.text}"
        for n, chunk in enumerate(chunks, start=1)
    )


def detect_language(*texts: str) -> str:
    """Guess "Turkish" or "English" from the first text that gives a clear signal.

    A text is Turkish if it has a Turkish letter or a common Turkish word (also in
    ASCII-folded form, e.g. "degil"), English if it has a common English word; texts
    with no signal are skipped. Defaults to "English".
    """
    for text in texts:
        if TURKISH_CHARS.intersection(text):
            return "Turkish"
        words = {word.casefold() for word in _WORD.findall(text)}
        if words & TURKISH_WORDS:
            return "Turkish"
        if words & ENGLISH_WORDS:
            return "English"
    return "English"
