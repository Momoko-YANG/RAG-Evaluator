"""Default RAG answer prompt, for generating answers from fixed contexts.

``RAG_SYSTEM_PROMPT`` + a user message with a ``<context>`` block and citation
instructions.  To measure your own app, pass its system prompt (``system_prompt`` /
``--system-prompt-file``) or keep this module in sync with the app's prompt; every
generation run records :func:`prompt_sha256` so that a changed prompt is visible in
the reports.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from typing import Any

RAG_SYSTEM_PROMPT = (
    "Answer using only the information in the content provided by upload or search. "
    'If the content does not contain the answer, reply "No information found." '
    "Do not guess or fill gaps with general knowledge."
)

_CITE_ALWAYS = (
    "- You must always include a citation at the end of your answer "
    "and don't include full stop after the citations."
)
_CONTROL_TAGS = re.compile(r"</?(context|memory|recall)>", re.I)
_CITATION_MARKUP = re.compile(r"\{%\s*citation[\s\S]*?/%\}")


def sanitize_control_tags(value: str) -> str:
    return _CONTROL_TAGS.sub(lambda m: m.group(0).replace("<", "&lt;").replace(">", "&gt;"), value)


def build_rag_context(documents: Sequence[dict[str, Any]]) -> str:
    return "\n------\n".join(
        f"[{index}]. file name: {sanitize_control_tags(str(d.get('title') or ''))} \n"
        f" file id: {d.get('id')} \n {sanitize_control_tags(str(d.get('content') or ''))}"
        for index, d in enumerate(documents)
    )


def build_rag_user_message(documents: Sequence[dict[str, Any]], question: str) -> str:
    citation = (
        '- Use the format for your citation {% citation items=[{name:"filename 1",id:"file id"}, '
        '{name:"filename 2",id:"file id"}] /%}'
    )
    return f"""
- Review the following content from documents uploaded by the user and create a final answer.
- If you don't know the answer, just say that you don't know. Don't try to make up an answer.
{_CITE_ALWAYS}
{citation}
{citation}

<context>
{build_rag_context(documents)}
</context>

question:
{question}
"""


def build_rag_messages(
    documents: Sequence[dict[str, Any]],
    question: str,
    *,
    persona: str = "",
    system_prompt: str | None = None,
) -> list[dict[str, str]]:
    """System prompt, persona, then the user message (no history or memory)."""
    return [
        {"role": "system", "content": system_prompt or RAG_SYSTEM_PROMPT},
        {"role": "system", "content": persona},
        {"role": "user", "content": build_rag_user_message(documents, question)},
    ]


def prompt_sha256(system_prompt: str | None = None, persona: str = "") -> str:
    """Identity of the prompt template (system prompt, persona, user-message template)."""
    template = build_rag_user_message(
        [{"title": "{title}", "id": "{id}", "content": "{content}"}], "{question}"
    )
    text = f"{system_prompt or RAG_SYSTEM_PROMPT}\n---\n{persona}\n---\n{template}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def strip_citation_markup(text: str) -> str:
    return _CITATION_MARKUP.sub("", text).strip()
