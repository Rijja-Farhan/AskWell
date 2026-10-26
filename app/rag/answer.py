"""Answer a question: retrieve top-k chunks, then ask the LLM to answer from them.

The system prompt is deliberately minimal: no citation, refusal or injection rules yet.
Phases 9 and 13 add those and must show a measurable gain over this baseline.
"""

from pydantic import BaseModel

from app.config import Settings
from app.rag.embeddings import Embedder
from app.rag.filters import RetrievalFilter
from app.rag.llm import LLM
from app.rag.rerank import Reranker
from app.rag.retrieve import RetrievedChunk, retrieve

SYSTEM_PROMPT = (
    "You are Askwell, the internal knowledge assistant for Nimbus Logistics. "
    "Answer the employee's question using the provided context."
)


class Answer(BaseModel):
    question: str
    answer: str
    chunks: list[RetrievedChunk]
    model: str


def build_user_message(question: str, chunks: list[RetrievedChunk]) -> str:
    context = "\n\n".join(f"[{c.doc_id}] {c.metadata.get('title', '')}\n{c.text}" for c in chunks)
    return f"Context:\n\n{context}\n\nQuestion: {question}"


def answer_question(
    question: str,
    k: int,
    embedder: Embedder,
    llm: LLM,
    settings: Settings,
    reranker: Reranker | None = None,
    filters: RetrievalFilter | None = None,
) -> Answer:
    chunks = retrieve(question, k, embedder, settings, reranker, filters)
    return answer_from_chunks(question, chunks, llm)


def answer_from_chunks(question: str, chunks: list[RetrievedChunk], llm: LLM) -> Answer:
    """Answer from chunks that were already retrieved (the eval runner reuses its retrieval)."""
    text = llm.complete(SYSTEM_PROMPT, build_user_message(question, chunks))
    return Answer(question=question, answer=text, chunks=chunks, model=llm.model)
