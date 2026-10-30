"""Role-based access: user profiles and the set of documents a user may read.

The rule lives in `app/dataset/schemas.py::level_readable` (shared with `can_access`); this
module applies it to index metadata. Fails closed: a chunk without `access_level` counts as
restricted, and a bad users file yields no users (the API then serves only the guest).

`GUEST` is the least-privilege profile used when a request names no user.
"""

import json
import logging
from pathlib import Path

from app.config import Settings
from app.dataset.schemas import UserProfile, level_readable
from app.rag.chunking import Scalar
from app.rag.store import resolve

log = logging.getLogger("askwell.access")

ACCESS_VERSION = 1  # bump when the access rule or its enforcement changes

GUEST = UserProfile(
    user_id="guest",
    name="Guest",
    role="Guest",
    department="HR",  # required by the schema; grants nothing (departments is empty)
    clearance="public",
    departments=[],
)


def load_users(settings: Settings) -> dict[str, UserProfile]:
    """Profiles by id from `<data_metadata_dir>/users.json`. Raises on a missing or invalid
    file: callers decide how to fail closed."""
    path: Path = resolve(settings.data_metadata_dir) / "users.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    users = [UserProfile.model_validate(item) for item in raw]
    return {u.user_id: u for u in users}


def can_read(user: UserProfile, metadata: dict[str, Scalar]) -> bool:
    """Whether `user` may read the document a chunk's metadata describes."""
    department = metadata.get("department")
    return level_readable(
        str(metadata.get("access_level") or "restricted"),  # missing level: restricted
        str(department) if department else None,
        user.clearance,
        list(user.departments),
    )


def accessible_doc_ids(user: UserProfile, metadatas: list[dict[str, Scalar]]) -> set[str]:
    """The doc ids (from chunk metadata) the user may read. A document is readable only if
    every one of its chunks is (fail closed on inconsistent metadata)."""
    verdict: dict[str, bool] = {}
    for m in metadatas:
        if "doc_id" in m:
            doc_id = str(m["doc_id"])
            verdict[doc_id] = verdict.get(doc_id, True) and can_read(user, m)
    return {d for d, ok in verdict.items() if ok}
