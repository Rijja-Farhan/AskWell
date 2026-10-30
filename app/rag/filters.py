"""Retrieval-time metadata filters: department, date and document version.

Policies and handbooks are revised; the old versions stay in the corpus so history can be
answered, but by default an employee must get the current rule. A filter is resolved to the
set of documents it allows, from the chunk metadata already stored in the index, and both
the dense query and BM25 rank only chunks of those documents.

- departments: keep only documents of these departments.
- as_of: per family, the version in force on that date (highest version whose effective date
  is on or before it); documents not yet effective are dropped. Replaces the version policy.
- version_policy: "active_only" drops superseded documents; "prefer_active" keeps them but
  never above an active chunk of the same family; "all" keeps everything.

- user: the requesting user's access (clearance and departments); only documents the user may
  read are ranked. It is intersected with every other constraint, so `departments` can narrow
  a user's results but never widen them.

This is the single place for retrieval-time constraints. Denied chunks are removed before
dense search, BM25, fusion and reranking, so they never reach the LLM prompt.
"""

import re
from datetime import date
from typing import Literal

from pydantic import BaseModel

from app.dataset.schemas import Department, UserProfile
from app.rag.access import accessible_doc_ids
from app.rag.chunking import Scalar

# Bump when the meaning of a filter or policy changes, so old eval runs no longer resume.
FILTER_VERSION = 2  # 2: access control (user) added

VersionPolicy = Literal["active_only", "prefer_active", "all"]
VERSION_SUFFIX = re.compile(r"-v\d+$")


class RetrievalFilter(BaseModel):
    departments: list[Department] | None = None
    as_of: date | None = None
    version_policy: VersionPolicy | None = None  # None means settings.version_policy
    user: UserProfile | None = None  # None means no access filtering


def family(metadata: dict[str, Scalar]) -> str:
    """The document family (all versions of one document); the doc id without -vN when the
    chunk has no family_id (the Phase 3 baseline collection)."""
    return str(metadata.get("family_id") or VERSION_SUFFIX.sub("", str(metadata["doc_id"])))


def is_superseded(metadata: dict[str, Scalar]) -> bool:
    return metadata.get("status") == "superseded"


def effective_policy(filters: RetrievalFilter | None, default: VersionPolicy) -> VersionPolicy:
    """The version policy that applies; "all" when as_of picks the versions instead."""
    filters = filters or RetrievalFilter()
    if filters.as_of is not None:
        return "all"
    return filters.version_policy or default


def allowed_doc_ids(
    metadatas: list[dict[str, Scalar]], filters: RetrievalFilter | None, policy: VersionPolicy
) -> set[str] | None:
    """The doc ids the filter keeps, from chunk metadata; None when it keeps everything."""
    filters = filters or RetrievalFilter()
    if (
        filters.user is None
        and not filters.departments
        and filters.as_of is None
        and policy != "active_only"
    ):
        return None
    docs = {str(m["doc_id"]): m for m in metadatas if "doc_id" in m}  # one entry per document
    if filters.user is not None:  # first, so later steps (as_of) only see readable documents
        allowed = accessible_doc_ids(filters.user, metadatas)
        docs = {d: m for d, m in docs.items() if d in allowed}
    if filters.departments:
        docs = {d: m for d, m in docs.items() if m.get("department") in filters.departments}
    if policy == "active_only":
        docs = {d: m for d, m in docs.items() if not is_superseded(m)}
    if filters.as_of is not None:
        cutoff = filters.as_of.isoformat()  # ISO dates compare correctly as strings
        in_force: dict[str, tuple[int, str]] = {}  # family -> (version, doc_id)
        for doc_id, m in docs.items():
            if str(m.get("effective_date", "")) <= cutoff:
                candidate = (int(m.get("version", 1)), doc_id)
                in_force[family(m)] = max(in_force.get(family(m), candidate), candidate)
        docs = {doc_id: docs[doc_id] for _, doc_id in in_force.values()}
    return set(docs)


def prefer_active(chunks: list) -> list:
    """Move each superseded chunk directly after the last active chunk of its family, keeping
    every other relative order. A superseded chunk with no active family chunk below it stays
    where it is. Ranks are not renumbered here."""
    last_active: dict[str, int] = {}
    for i, c in enumerate(chunks):
        if not is_superseded(c.metadata):
            last_active[family(c.metadata)] = i
    ordered, deferred = [], {}
    for i, c in enumerate(chunks):
        fam = family(c.metadata)
        if is_superseded(c.metadata) and last_active.get(fam, -1) > i:
            deferred.setdefault(fam, []).append(c)
            continue
        ordered.append(c)
        if last_active.get(fam) == i:
            ordered += deferred.pop(fam, [])
    return ordered
