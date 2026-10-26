"""Pydantic contracts for the synthetic dataset.

Later phases read these fields by name: adding fields is fine, renaming or removing is not.
"""

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Department = Literal["HR", "IT", "Operations", "Finance"]
DocType = Literal["policy", "manual", "meeting_notes", "faq", "table"]
DocFormat = Literal["pdf", "docx", "md", "txt", "html", "csv"]
AccessLevel = Literal["public", "internal", "confidential", "restricted"]
Category = Literal[
    "simple",
    "exact-term",
    "multi-doc",
    "conflicting-versions",
    "unanswerable",
    "access-denied",
    "injection",
]
ExpectedBehavior = Literal["answer", "refuse_unknown", "refuse_access", "resist_injection"]

DOC_ID_PATTERN = r"^(hr|it|ops|fin)-(pol|man|mtg|faq|tbl)-\d{3}-v\d+$"

ACCESS_RANK: dict[str, int] = {"public": 0, "internal": 1, "confidential": 2, "restricted": 3}

# Folder under data/raw/ for each department.
DEPT_DIRS: dict[str, str] = {
    "HR": "hr",
    "IT": "it",
    "Operations": "operations",
    "Finance": "finance",
}


class DocumentMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(pattern=DOC_ID_PATTERN)
    family_id: str
    title: str
    department: Department
    doc_type: DocType
    format: DocFormat
    path: str
    version: int = Field(ge=1)
    status: Literal["active", "superseded"]
    effective_date: date
    last_updated: date
    supersedes: str | None = None
    superseded_by: str | None = None
    access_level: AccessLevel
    owner: str
    tags: list[str]
    key_terms: list[str]
    has_tables: bool
    contains_injection: bool
    # Extra fields (beyond the minimum contract), used by the validator.
    section_headings: list[str] = []
    max_table_rows: int = 0
    word_count: int = 0


class UserProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str
    name: str
    role: str
    department: Department
    clearance: AccessLevel
    departments: list[Department]


class EvalQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^q\d{3}$")
    question: str
    category: Category
    user_id: str
    gold_doc_ids: list[str]
    reference_answer: str
    expected_behavior: ExpectedBehavior
    exact_terms: list[str] = []
    notes: str = ""
    # Extra field: older versions a conflicting-versions question must NOT be answered from.
    superseded_doc_ids: list[str] = []
    # Extra field for injection questions: where the attack sits.
    attack_in: Literal["question", "document"] | None = None


def can_access(user: UserProfile, doc: DocumentMeta) -> bool:
    """Return True if `user` may read `doc`.

    Readable iff the doc's level is at or below the user's clearance, and either the level is
    public/internal or the doc's department is one the user is granted.
    """
    if ACCESS_RANK[doc.access_level] > ACCESS_RANK[user.clearance]:
        return False
    if doc.access_level in ("public", "internal"):
        return True
    return doc.department in user.departments
