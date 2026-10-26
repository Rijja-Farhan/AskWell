import json
import shutil
from pathlib import Path

import pytest

from app.config import Settings
from app.dataset.schemas import DocumentMeta, UserProfile, can_access
from app.dataset.validate import validate_dataset

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def settings(monkeypatch, tmp_path):
    """Default paths, unaffected by a local `.env` or shell variables."""
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
    monkeypatch.chdir(tmp_path)
    return Settings()


@pytest.fixture
def copy_root(tmp_path, settings):
    """A copy of the committed dataset under tmp_path, safe to corrupt."""
    root = tmp_path / "repo"
    for rel in (settings.data_raw_dir, settings.data_metadata_dir):
        shutil.copytree(ROOT / rel, root / rel)
    eval_path = root / settings.eval_set_path
    eval_path.parent.mkdir(parents=True)
    shutil.copy(ROOT / settings.eval_set_path, eval_path)
    return root


def read_json(path: Path) -> list:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: list) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def read_questions(root: Path, settings: Settings) -> list[dict]:
    lines = (root / settings.eval_set_path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def write_questions(root: Path, settings: Settings, questions: list[dict]) -> None:
    text = "\n".join(json.dumps(q) for q in questions) + "\n"
    (root / settings.eval_set_path).write_text(text, encoding="utf-8")


def assert_error(errors: list[str], fragment: str) -> None:
    assert any(fragment in e for e in errors), f"no error containing {fragment!r} in {errors}"


def test_committed_dataset_is_valid(settings):
    assert validate_dataset(ROOT, settings) == []


def test_copy_is_valid(copy_root, settings):
    assert validate_dataset(copy_root, settings) == []


def test_missing_file(copy_root, settings):
    (copy_root / settings.data_raw_dir / "hr" / "hr-pol-003-v1.pdf").unlink()
    assert_error(validate_dataset(copy_root, settings), "metadata entry has no file")


def test_unlisted_file(copy_root, settings):
    (copy_root / settings.data_raw_dir / "it" / "notes.txt").write_text("stray")
    assert_error(validate_dataset(copy_root, settings), "file has no metadata entry")


def test_duplicate_doc_id(copy_root, settings):
    path = copy_root / settings.data_metadata_dir / "documents.json"
    docs = read_json(path)
    write_json(path, [docs[0], *docs])
    assert_error(validate_dataset(copy_root, settings), f"duplicate doc_id {docs[0]['doc_id']}")


def test_wrong_category_count(copy_root, settings):
    questions = read_questions(copy_root, settings)
    simple = next(i for i, q in enumerate(questions) if q["category"] == "simple")
    del questions[simple]
    for n, q in enumerate(questions, start=1):
        q["id"] = f"q{n:03d}"
    write_questions(copy_root, settings, questions)
    errors = validate_dataset(copy_root, settings)
    assert_error(errors, "category simple: 15 questions, need 16")
    assert_error(errors, "need exactly 80")


def test_access_denied_gold_readable(copy_root, settings):
    questions = read_questions(copy_root, settings)
    q = next(q for q in questions if q["category"] == "access-denied")
    q["user_id"] = "u06"  # all-department executive can read everything
    write_questions(copy_root, settings, questions)
    assert_error(validate_dataset(copy_root, settings), "u06 CAN read gold doc")


def test_broken_supersession_chain(copy_root, settings):
    path = copy_root / settings.data_metadata_dir / "documents.json"
    docs = read_json(path)
    old = next(d for d in docs if d["status"] == "superseded")
    old["status"] = "active"
    write_json(path, docs)
    assert_error(validate_dataset(copy_root, settings), "active documents, need exactly 1")


def test_exact_term_missing_from_file(copy_root, settings):
    path = copy_root / settings.data_raw_dir / "it" / "it-tbl-001-v1.csv"
    path.write_text(path.read_text(encoding="utf-8").replace("NX-5002", "NX-5003"))
    assert_error(validate_dataset(copy_root, settings), "key term NX-5002 not found")


def user(clearance: str, departments: list[str]) -> UserProfile:
    return UserProfile(
        user_id="u",
        name="Test",
        role="Tester",
        department="HR",
        clearance=clearance,
        departments=departments,
    )


def doc(level: str, department: str) -> DocumentMeta:
    return DocumentMeta.model_construct(access_level=level, department=department)


@pytest.mark.parametrize(
    ("clearance", "departments", "level", "doc_dept", "expected"),
    [
        ("public", [], "public", "HR", True),
        ("public", [], "internal", "HR", False),
        ("internal", [], "internal", "IT", True),
        ("internal", ["HR"], "confidential", "HR", False),  # clearance too low
        ("confidential", ["HR"], "confidential", "HR", True),
        ("confidential", ["HR"], "confidential", "IT", False),  # wrong department
        ("confidential", ["HR"], "restricted", "HR", False),  # clearance too low
        ("confidential", ["HR"], "internal", "Finance", True),  # internal ignores department
        ("restricted", ["IT"], "restricted", "IT", True),
        ("restricted", ["IT"], "restricted", "Operations", False),
        ("restricted", ["IT"], "confidential", "IT", True),
        ("restricted", ["HR", "IT", "Operations", "Finance"], "restricted", "Finance", True),
    ],
)
def test_can_access(clearance, departments, level, doc_dept, expected):
    assert can_access(user(clearance, departments), doc(level, doc_dept)) is expected
