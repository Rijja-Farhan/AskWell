"""Access rule and retrieval filter: offline, on the real corpus with a fake embedder."""

import json

import pytest

from app.dataset.schemas import DocumentMeta, UserProfile, can_access
from app.rag.access import GUEST, accessible_doc_ids, can_read, load_users
from app.rag.bm25 import get_index
from app.rag.filters import RetrievalFilter, allowed_doc_ids
from app.rag.retrieve import retrieve
from app.rag.store import build_index, open_collection
from tests.conftest import ROOT, FakeEmbedder, make_settings


@pytest.fixture(scope="module")
def settings(corpus, tmp_path_factory):
    settings = make_settings(tmp_path_factory.mktemp("access_index"), retrieval_mode="bm25")
    build_index(corpus, FakeEmbedder(), settings)
    return settings


@pytest.fixture(scope="module")
def metadatas(settings):
    return get_index(open_collection(settings)).metadatas


@pytest.fixture(scope="module")
def users(settings):
    return load_users(settings)


@pytest.fixture(scope="module")
def docs() -> list[DocumentMeta]:
    raw = json.loads((ROOT / "data/metadata/documents.json").read_text(encoding="utf-8"))
    return [DocumentMeta.model_validate(d) for d in raw]


def test_load_users(users):
    assert len(users) == 10 and users["u05"].clearance == "public"
    assert users["u06"].clearance == "restricted" and len(users["u06"].departments) == 4


def test_rule_matches_can_access_for_every_user_and_document(users, docs, metadatas):
    assert len(docs) == 52
    for user in users.values():
        expected = {d.doc_id for d in docs if can_access(user, d)}
        assert accessible_doc_ids(user, metadatas) == expected, user.user_id


def test_guest_sees_public_documents_only(docs, metadatas):
    public = {d.doc_id for d in docs if d.access_level == "public"}
    assert len(public) == 8
    assert accessible_doc_ids(GUEST, metadatas) == public


def test_missing_access_level_is_restricted(users):
    chunk = {"doc_id": "x-1", "department": "HR"}
    assert not can_read(users["u02"], chunk)
    assert not can_read(users["u01"], chunk)  # confidential clearance is below restricted
    assert can_read(users["u06"], chunk)
    assert accessible_doc_ids(GUEST, [chunk]) == set()


def test_one_unreadable_chunk_hides_the_document():
    metas = [
        {"doc_id": "d1", "access_level": "public", "department": "HR"},
        {"doc_id": "d1", "department": "HR"},  # no level: restricted
    ]
    assert accessible_doc_ids(GUEST, metas) == set()


def test_filter_intersects_with_departments(users, docs, metadatas):
    u01 = users["u01"]  # confidential, HR only
    readable = accessible_doc_ids(u01, metadatas)
    got = allowed_doc_ids(metadatas, RetrievalFilter(user=u01, departments=["IT"]), "all")
    assert got is not None and got <= readable
    assert got == {d.doc_id for d in docs if d.doc_id in readable and d.department == "IT"}
    assert not any(
        d.access_level in ("confidential", "restricted") for d in docs if d.doc_id in got
    )


def test_departments_cannot_widen_access(users, docs, metadatas):
    u05 = users["u05"]
    got = allowed_doc_ids(
        metadatas,
        RetrievalFilter(user=u05, departments=["HR", "IT", "Operations", "Finance"]),
        "all",
    )
    assert got == {d.doc_id for d in docs if d.access_level == "public"}


def test_filter_without_user_keeps_everything(metadatas):
    assert allowed_doc_ids(metadatas, RetrievalFilter(), "all") is None


def test_filter_composes_with_version_policy(users, docs, metadatas):
    got = allowed_doc_ids(metadatas, RetrievalFilter(user=users["u06"]), "active_only")
    assert got == {d.doc_id for d in docs if d.status == "active"}


@pytest.mark.parametrize("uid", ["u01", "u02", "u05", "u06", "guest"])
def test_retrieval_returns_only_readable_chunks(uid, users, docs, metadatas, settings):
    user = GUEST if uid == "guest" else users[uid]
    readable = accessible_doc_ids(user, metadatas)
    for doc in docs:
        query = f"{doc.title} {' '.join(doc.key_terms)}"
        hits = retrieve(query, 10, FakeEmbedder(), settings, None, RetrievalFilter(user=user))
        assert {h.doc_id for h in hits} <= readable


def test_hybrid_and_dense_modes_filter_too(users, metadatas, settings):
    readable = accessible_doc_ids(users["u05"], metadatas)
    for mode in ("dense", "hybrid"):
        s = settings.model_copy(update={"retrieval_mode": mode})
        hits = retrieve(
            "What is the salary range for grade G5?",
            20,
            FakeEmbedder(),
            s,
            None,
            RetrievalFilter(user=users["u05"]),
        )
        assert hits and {h.doc_id for h in hits} <= readable


def test_empty_access_set_returns_nothing(settings):
    nobody = UserProfile(
        user_id="x", name="x", role="x", department="HR", clearance="public", departments=[]
    )
    only_restricted = [{"doc_id": "d", "access_level": "restricted", "department": "HR"}]
    assert allowed_doc_ids(only_restricted, RetrievalFilter(user=nobody), "all") == set()


def test_bad_users_file_fails(tmp_path):
    (tmp_path / "users.json").write_text("not json")
    with pytest.raises(ValueError):
        load_users(make_settings(tmp_path, data_metadata_dir=tmp_path))
