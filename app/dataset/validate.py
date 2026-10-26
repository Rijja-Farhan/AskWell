"""Validate the synthetic dataset: corpus files, document metadata, users and the eval set.

`validate_dataset()` returns a list of error messages; an empty list means the dataset is valid.
`dataset_summary()` returns the coverage counts printed by `scripts/validate_dataset.py`.
"""

import csv
import html
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from app.config import Settings, get_settings
from app.dataset.schemas import (
    ACCESS_RANK,
    DEPT_DIRS,
    DocumentMeta,
    EvalQuestion,
    UserProfile,
    can_access,
)

TEXT_FORMATS = {"md", "txt", "html", "csv"}
PREFIX_DEPT = {"hr": "HR", "it": "IT", "ops": "Operations", "fin": "Finance"}
PREFIX_TYPE = {
    "pol": "policy",
    "man": "manual",
    "mtg": "meeting_notes",
    "faq": "faq",
    "tbl": "table",
}

# Exact-match token kinds that must appear somewhere in the corpus.
KEY_TERM_PATTERNS = {
    "policy code": re.compile(r"^(HR|IT|OPS|FIN)-(POL|MAN|MTG|FAQ|TBL)-\d{3}$"),
    "IT error code": re.compile(r"^NX-\d{4}$"),
    "GL account": re.compile(r"^GL-\d{4}$"),
    "site code": re.compile(r"^WH-[A-Z]{3}-\d{2}$"),
}
INJECTION_PATTERN = re.compile(r"ignore (all |any )?(previous|prior|above) instructions", re.I)

MIN_DOCS = 48
MIN_PER_DEPT = 10
# (doc_type, format, department or None) -> minimum count
FORMAT_MINIMUMS = {
    ("policy", "pdf", None): 8,
    ("policy", "docx", None): 6,
    ("manual", "md", "IT"): 8,
    ("meeting_notes", "txt", None): 4,
    ("meeting_notes", "md", None): 2,
    ("faq", "html", None): 6,
    ("table", "csv", None): 4,
    ("table", "pdf", None): 3,
}
MIN_FAMILIES = 6
MIN_FAMILY_DEPTS = 3
MIN_INJECTION_DOCS = 3
MIN_PER_ACCESS_LEVEL = 4
MIN_USERS = 8
CATEGORY_COUNTS = {
    "simple": 16,
    "exact-term": 12,
    "multi-doc": 12,
    "conflicting-versions": 10,
    "unanswerable": 10,
    "access-denied": 10,
    "injection": 10,
}
EXPECTED_BEHAVIOR = {
    "simple": "answer",
    "exact-term": "answer",
    "multi-doc": "answer",
    "conflicting-versions": "answer",
    "unanswerable": "refuse_unknown",
    "access-denied": "refuse_access",
    "injection": "resist_injection",
}
ANSWERABLE = {"simple", "exact-term", "multi-doc", "conflicting-versions"}
MIN_INJECTION_IN_QUESTION = 4
MIN_INJECTION_IN_DOCUMENT = 3


@dataclass
class Dataset:
    root: Path
    settings: Settings
    docs: list[DocumentMeta] = field(default_factory=list)
    users: list[UserProfile] = field(default_factory=list)
    questions: list[EvalQuestion] = field(default_factory=list)

    @property
    def raw_dir(self) -> Path:
        return self.root / self.settings.data_raw_dir

    @property
    def metadata_dir(self) -> Path:
        return self.root / self.settings.data_metadata_dir

    @property
    def eval_path(self) -> Path:
        return self.root / self.settings.eval_set_path


def read_doc_text(root: Path, doc: DocumentMeta) -> str:
    """Return the file text for text formats (HTML entities decoded), else ''."""
    if doc.format not in TEXT_FORMATS:
        return ""
    text = (root / doc.path).read_text(encoding="utf-8")
    return html.unescape(text) if doc.format == "html" else text


def load_dataset(root: Path, settings: Settings) -> tuple[Dataset, list[str]]:
    """Parse the three metadata/eval files into models. Returns the dataset and parse errors."""
    ds = Dataset(root=root, settings=settings)
    errors: list[str] = []

    def load_json(path: Path) -> list:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{path}: cannot read JSON ({exc})")
            return []
        if not isinstance(data, list):
            errors.append(f"{path}: expected a JSON array")
            return []
        return data

    for i, raw in enumerate(load_json(ds.metadata_dir / "documents.json")):
        try:
            ds.docs.append(DocumentMeta.model_validate(raw))
        except ValidationError as exc:
            errors.append(f"documents.json[{i}]: {exc}")
    for i, raw in enumerate(load_json(ds.metadata_dir / "users.json")):
        try:
            ds.users.append(UserProfile.model_validate(raw))
        except ValidationError as exc:
            errors.append(f"users.json[{i}]: {exc}")
    try:
        lines = ds.eval_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        errors.append(f"{ds.eval_path}: cannot read ({exc})")
        lines = []
    for n, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            ds.questions.append(EvalQuestion.model_validate_json(line))
        except ValidationError as exc:
            errors.append(f"eval line {n}: {exc}")
    return ds, errors


def _check_documents(ds: Dataset) -> list[str]:
    errors: list[str] = []
    docs = ds.docs
    raw_rel = ds.settings.data_raw_dir.as_posix()

    ids = [d.doc_id for d in docs]
    for doc_id, n in Counter(ids).items():
        if n > 1:
            errors.append(f"duplicate doc_id {doc_id} ({n} entries)")
    if ids != sorted(ids):
        errors.append("documents.json is not sorted by doc_id")

    # Files on disk <-> metadata entries, one to one.
    on_disk = {
        p.relative_to(ds.root).as_posix()
        for p in ds.raw_dir.rglob("*")
        if p.is_file() and p.name != ".gitkeep"
    }
    listed = {d.path for d in docs}
    for path in sorted(on_disk - listed):
        errors.append(f"file has no metadata entry: {path}")
    for path in sorted(listed - on_disk):
        errors.append(f"metadata entry has no file: {path}")

    for d in docs:
        prefix, type_code = d.doc_id.split("-")[:2]
        if PREFIX_DEPT[prefix] != d.department:
            errors.append(f"{d.doc_id}: department {d.department} does not match id prefix")
        if PREFIX_TYPE[type_code] != d.doc_type:
            errors.append(f"{d.doc_id}: doc_type {d.doc_type} does not match id")
        if d.family_id != d.doc_id.rsplit("-v", 1)[0]:
            errors.append(f"{d.doc_id}: family_id {d.family_id} should be the id without -vN")
        if int(d.doc_id.rsplit("-v", 1)[1]) != d.version:
            errors.append(f"{d.doc_id}: version {d.version} does not match id")
        expected_path = f"{raw_rel}/{DEPT_DIRS[d.department]}/{d.doc_id}.{d.format}"
        if d.path != expected_path:
            errors.append(f"{d.doc_id}: path {d.path} should be {expected_path}")
        for day in (d.effective_date, d.last_updated):
            if not 2023 <= day.year <= 2026:
                errors.append(f"{d.doc_id}: date {day} outside 2023-2026")
        if d.doc_type in ("policy", "manual", "faq") and len(d.section_headings) < 3:
            errors.append(f"{d.doc_id}: needs >= 3 headed sections")
        if d.doc_type == "table" and (not d.has_tables or d.max_table_rows < 5):
            errors.append(f"{d.doc_id}: table doc needs a table with >= 5 rows")
        if d.format == "csv" and d.path in on_disk:
            with (ds.root / d.path).open(encoding="utf-8", newline="") as f:
                rows = list(csv.reader(f))
            if len(rows) - 1 < 5:
                errors.append(f"{d.doc_id}: CSV has fewer than 5 data rows")
        if d.format in TEXT_FORMATS and d.path in on_disk:
            text = read_doc_text(ds.root, d)
            for term in d.key_terms:
                if term not in text:
                    errors.append(f"{d.doc_id}: key term {term} not found in file")
            for heading in d.section_headings:
                if heading not in text:
                    errors.append(f"{d.doc_id}: section heading {heading!r} not found in file")
            if d.contains_injection and not INJECTION_PATTERN.search(text):
                errors.append(f"{d.doc_id}: contains_injection but no injection text found")

    # Corpus-wide coverage.
    if len(docs) < MIN_DOCS:
        errors.append(f"corpus has {len(docs)} documents, need >= {MIN_DOCS}")
    per_dept = Counter(d.department for d in docs)
    for dept in DEPT_DIRS:
        if per_dept[dept] < MIN_PER_DEPT:
            errors.append(f"{dept} has {per_dept[dept]} documents, need >= {MIN_PER_DEPT}")
    for (doc_type, fmt, dept), minimum in FORMAT_MINIMUMS.items():
        n = sum(
            1
            for d in docs
            if d.doc_type == doc_type and d.format == fmt and dept in (None, d.department)
        )
        if n < minimum:
            label = f"{dept + ' ' if dept else ''}{doc_type} {fmt}"
            errors.append(f"{label}: {n} documents, need >= {minimum}")
    all_terms = {t for d in docs for t in d.key_terms}
    for kind, pattern in KEY_TERM_PATTERNS.items():
        if not any(pattern.match(t) for t in all_terms):
            errors.append(f"no {kind} found in any document's key_terms")
    per_level = Counter(d.access_level for d in docs)
    for level in ACCESS_RANK:
        if per_level[level] < MIN_PER_ACCESS_LEVEL:
            errors.append(f"access level {level}: {per_level[level]} documents, need >= 4")
    for dept in DEPT_DIRS:
        if not any(
            d.department == dept and d.access_level in ("confidential", "restricted") for d in docs
        ):
            errors.append(f"{dept} has no confidential or restricted document")
    n_injection = sum(d.contains_injection for d in docs)
    if n_injection < MIN_INJECTION_DOCS:
        errors.append(f"{n_injection} injection documents, need >= {MIN_INJECTION_DOCS}")

    errors += _check_families(docs)
    return errors


def _check_families(docs: list[DocumentMeta]) -> list[str]:
    errors: list[str] = []
    by_id = {d.doc_id: d for d in docs}
    families: dict[str, list[DocumentMeta]] = defaultdict(list)
    for d in docs:
        families[d.family_id].append(d)

    versioned_depts = set()
    n_versioned = 0
    for family_id, members in families.items():
        members.sort(key=lambda d: d.version)
        active = [d for d in members if d.status == "active"]
        if len(active) != 1:
            errors.append(f"family {family_id}: {len(active)} active documents, need exactly 1")
        elif active[0] is not members[-1]:
            errors.append(f"family {family_id}: active doc is not the highest version")
        if len({d.version for d in members}) != len(members):
            errors.append(f"family {family_id}: duplicate versions")
        for d in members:
            if d.status == "superseded":
                newer = by_id.get(d.superseded_by or "")
                if newer is None:
                    errors.append(f"{d.doc_id}: superseded but superseded_by is missing/unknown")
                elif (
                    newer.family_id != family_id
                    or newer.version <= d.version
                    or newer.effective_date <= d.effective_date
                ):
                    errors.append(
                        f"{d.doc_id}: superseded_by {newer.doc_id} must be a later, "
                        "higher-version doc in the same family"
                    )
            elif d.superseded_by is not None:
                errors.append(f"{d.doc_id}: active doc must have superseded_by = null")
            if d.supersedes is not None:
                older = by_id.get(d.supersedes)
                if older is None or older.superseded_by != d.doc_id:
                    errors.append(f"{d.doc_id}: supersedes {d.supersedes} does not link back")
        if len(members) > 1:
            n_versioned += 1
            versioned_depts.add(members[0].department)

    if n_versioned < MIN_FAMILIES:
        errors.append(f"{n_versioned} version families, need >= {MIN_FAMILIES}")
    if len(versioned_depts) < MIN_FAMILY_DEPTS:
        errors.append(
            f"version families span {len(versioned_depts)} departments, need >= {MIN_FAMILY_DEPTS}"
        )
    return errors


def _check_users(users: list[UserProfile]) -> list[str]:
    errors: list[str] = []
    if len(users) < MIN_USERS:
        errors.append(f"{len(users)} users, need >= {MIN_USERS}")
    for user_id, n in Counter(u.user_id for u in users).items():
        if n > 1:
            errors.append(f"duplicate user_id {user_id}")
    missing_depts = set(DEPT_DIRS) - {u.department for u in users}
    if missing_depts:
        errors.append(f"no user in departments: {sorted(missing_depts)}")
    missing_levels = set(ACCESS_RANK) - {u.clearance for u in users}
    if missing_levels:
        errors.append(f"no user with clearance: {sorted(missing_levels)}")
    if not any(u.clearance == "public" for u in users):
        errors.append("no public-only contractor")
    if not any(u.clearance == "restricted" and set(u.departments) == set(DEPT_DIRS) for u in users):
        errors.append("no all-department executive (restricted clearance, all departments)")
    return errors


def _check_questions(ds: Dataset) -> list[str]:
    errors: list[str] = []
    qs = ds.questions
    docs = {d.doc_id: d for d in ds.docs}
    users = {u.user_id: u for u in ds.users}

    if len(qs) != 80:
        errors.append(f"eval set has {len(qs)} questions, need exactly 80")
    expected_ids = [f"q{i:03d}" for i in range(1, len(qs) + 1)]
    if [q.id for q in qs] != expected_ids:
        errors.append("question ids must run q001, q002, ... in order without gaps")
    counts = Counter(q.category for q in qs)
    for category, need in CATEGORY_COUNTS.items():
        if counts[category] != need:
            errors.append(f"category {category}: {counts[category]} questions, need {need}")

    n_attack_question = 0
    n_attack_document = 0
    for q in qs:
        where = f"{q.id} ({q.category})"
        if q.expected_behavior != EXPECTED_BEHAVIOR[q.category]:
            errors.append(f"{where}: expected_behavior should be {EXPECTED_BEHAVIOR[q.category]}")
        user = users.get(q.user_id)
        if user is None:
            errors.append(f"{where}: unknown user_id {q.user_id}")
        unknown = [g for g in q.gold_doc_ids + q.superseded_doc_ids if g not in docs]
        if unknown:
            errors.append(f"{where}: unknown doc ids {unknown}")
            continue
        gold = [docs[g] for g in q.gold_doc_ids]
        if len(set(q.gold_doc_ids)) != len(q.gold_doc_ids):
            errors.append(f"{where}: duplicate gold doc ids")

        if q.category in ANSWERABLE:
            if not gold:
                errors.append(f"{where}: needs >= 1 gold doc")
            if not q.reference_answer.strip():
                errors.append(f"{where}: needs a reference answer")
        if q.category == "multi-doc" and len({d.family_id for d in gold}) < 2:
            errors.append(f"{where}: needs gold docs from >= 2 different documents")
        if q.category == "exact-term" and not q.exact_terms:
            errors.append(f"{where}: needs >= 1 exact term")
        if q.category == "conflicting-versions":
            if not q.superseded_doc_ids:
                errors.append(f"{where}: needs superseded_doc_ids")
            for d in gold:
                has_old = any(
                    o.family_id == d.family_id and o.status == "superseded" for o in ds.docs
                )
                if d.status != "active" or not has_old:
                    errors.append(
                        f"{where}: gold {d.doc_id} must be the active doc of a versioned family"
                    )
            gold_families = {d.family_id for d in gold}
            for s in q.superseded_doc_ids:
                if docs[s].status != "superseded" or docs[s].family_id not in gold_families:
                    errors.append(f"{where}: {s} is not a superseded version of a gold doc")
        if q.category == "unanswerable" and gold:
            errors.append(f"{where}: unanswerable questions must have no gold docs")
        if q.category == "access-denied" and not gold:
            errors.append(f"{where}: needs >= 1 gold doc the user cannot read")
        if q.category == "injection":
            if q.attack_in == "question":
                n_attack_question += 1
            elif q.attack_in == "document":
                if not any(d.contains_injection for d in gold):
                    errors.append(f"{where}: attack_in=document but no gold doc has an injection")
                else:
                    n_attack_document += 1
            else:
                errors.append(f"{where}: injection questions need attack_in")

        if user is not None:
            for d in gold:
                readable = can_access(user, d)
                if q.category == "access-denied" and readable:
                    errors.append(f"{where}: {q.user_id} CAN read gold doc {d.doc_id}")
                if q.category != "access-denied" and not readable:
                    errors.append(f"{where}: {q.user_id} cannot read gold doc {d.doc_id}")

        for term in q.exact_terms:
            holders = [d for d in gold if term in d.key_terms]
            if not holders:
                errors.append(f"{where}: exact term {term} not in any gold doc's key_terms")
            for d in holders:
                if d.format in TEXT_FORMATS and term not in read_doc_text(ds.root, d):
                    errors.append(f"{where}: exact term {term} not found in {d.path}")

    if n_attack_question < MIN_INJECTION_IN_QUESTION:
        errors.append(
            f"{n_attack_question} injection questions attack via the question, "
            f"need >= {MIN_INJECTION_IN_QUESTION}"
        )
    if n_attack_document < MIN_INJECTION_IN_DOCUMENT:
        errors.append(
            f"{n_attack_document} injection questions target an injected document, "
            f"need >= {MIN_INJECTION_IN_DOCUMENT}"
        )
    return errors


def validate_dataset(root: Path = Path("."), settings: Settings | None = None) -> list[str]:
    """Check the whole dataset under `root`. Returns error messages (empty = valid)."""
    ds, errors = load_dataset(root, settings or get_settings())
    errors += _check_documents(ds)
    errors += _check_users(ds.users)
    errors += _check_questions(ds)
    return errors


def dataset_summary(root: Path = Path("."), settings: Settings | None = None) -> dict:
    """Coverage counts for the CLI summary."""
    ds, _ = load_dataset(root, settings or get_settings())
    families = Counter(d.family_id for d in ds.docs)
    return {
        "documents": len(ds.docs),
        "by_department": dict(sorted(Counter(d.department for d in ds.docs).items())),
        "by_format": dict(sorted(Counter(d.format for d in ds.docs).items())),
        "by_type_format": dict(
            sorted(Counter(f"{d.doc_type}/{d.format}" for d in ds.docs).items())
        ),
        "by_access_level": {
            level: sum(d.access_level == level for d in ds.docs) for level in ACCESS_RANK
        },
        "version_families": sum(n > 1 for n in families.values()),
        "injection_documents": sum(d.contains_injection for d in ds.docs),
        "words_total": sum(d.word_count for d in ds.docs),
        "users": len(ds.users),
        "users_by_clearance": {
            level: sum(u.clearance == level for u in ds.users) for level in ACCESS_RANK
        },
        "questions": len(ds.questions),
        "by_category": {c: sum(q.category == c for q in ds.questions) for c in CATEGORY_COUNTS},
    }
