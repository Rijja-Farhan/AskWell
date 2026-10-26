"""Second-stage reranking: rescore first-stage candidates with a cross-encoder.

A cross-encoder reads the question and one chunk together and outputs a relevance score, so
it sees word interactions that embeddings (scored separately) and BM25 (word counts) miss.
It is too slow to run over the whole index, so it only rescores the top `rerank_candidates`
from dense, BM25 or hybrid retrieval. The model runs locally on ONNX Runtime (CPU): no API
quota, and the same scores on every run.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from app.config import Settings
from app.rag.store import resolve

if TYPE_CHECKING:  # retrieve imports this module
    from app.rag.retrieve import RetrievedChunk

# Bump when the reranker input changes (text formatting, truncation), so old eval runs
# no longer resume against the new behavior.
RERANK_VERSION = 1
MAX_LENGTH = 512  # tokens per (question, chunk) pair; both supported models allow 512
BATCH_SIZE = 16


class Reranker(Protocol):
    model: str

    def score(self, question: str, texts: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    """A Hugging Face cross-encoder exported to ONNX, pinned to one repo revision.

    Higher scores mean more relevant. Scores are raw logits: comparable within one question,
    not calibrated across questions.
    """

    def __init__(
        self, model: str, revision: str, onnx_file: str, cache_dir: Path, threads: int = 0
    ) -> None:
        self.model = model

        def fetch(filename: str) -> str:
            return hf_hub_download(model, filename, revision=revision, cache_dir=cache_dir)

        self.tokenizer = Tokenizer.from_file(fetch("tokenizer.json"))
        # Truncate only the chunk side, so the whole question always reaches the model.
        self.tokenizer.enable_truncation(MAX_LENGTH, strategy="only_second")
        self.tokenizer.enable_padding()
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads  # 0 = ONNX Runtime picks (all cores)
        self.session = ort.InferenceSession(
            fetch(onnx_file), options, providers=["CPUExecutionProvider"]
        )
        self.input_names = {i.name for i in self.session.get_inputs()}
        self.score("warm up", ["warm up"])  # the first run allocates; keep it out of latency

    def score(self, question: str, texts: list[str]) -> list[float]:
        scores: list[float] = []
        for start in range(0, len(texts), BATCH_SIZE):
            batch = texts[start : start + BATCH_SIZE]
            scores += self._logits(self.tokenizer.encode_batch([(question, t) for t in batch]))
        return scores

    def _logits(self, encodings) -> list[float]:
        features = {
            "input_ids": [e.ids for e in encodings],
            "attention_mask": [e.attention_mask for e in encodings],
            "token_type_ids": [e.type_ids for e in encodings],  # BERT only; XLM-R has none
        }
        feeds = {
            name: np.array(values, dtype=np.int64)
            for name, values in features.items()
            if name in self.input_names
        }
        [logits] = self.session.run(None, feeds)
        return [float(v) for v in logits[:, 0]]  # one relevance logit per pair


def get_reranker(settings: Settings) -> Reranker | None:
    """Load the configured reranker (downloading it once), or None when reranking is off."""
    if settings.reranker == "none":
        return None
    return CrossEncoderReranker(
        settings.rerank_model,
        settings.rerank_model_revision,
        settings.rerank_onnx_file,
        resolve(settings.rerank_model_dir),
    )


def rerank(
    question: str, candidates: list[RetrievedChunk], reranker: Reranker, k: int
) -> list[RetrievedChunk]:
    """Order candidates by reranker score (ties keep first-stage order) and keep the top k.
    Each result's `score` becomes the reranker score and `first_stage_rank` its old rank."""
    if not candidates:
        return []
    scores = reranker.score(question, [c.text for c in candidates])
    order = sorted(range(len(candidates)), key=lambda i: (-scores[i], candidates[i].rank))[:k]
    return [
        candidates[i].model_copy(
            update={"rank": rank, "score": scores[i], "first_stage_rank": candidates[i].rank}
        )
        for rank, i in enumerate(order, start=1)
    ]
