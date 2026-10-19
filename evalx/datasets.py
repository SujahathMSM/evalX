"""Loading golden datasets from YAML."""

from pathlib import Path

import yaml
from pydantic import ValidationError

from evalx.schemas import TestCase


class DatasetError(Exception):
    """Raised when a dataset file is missing or invalid."""


def load_dataset(path: str | Path) -> list[TestCase]:
    """Load and validate a golden dataset.

    Accepts a directory (loads ``cases.yaml`` inside it) or a YAML file
    containing a list of test cases.
    """
    p = Path(path)
    if p.is_dir():
        p = p / "cases.yaml"
    if not p.exists():
        raise DatasetError(f"dataset not found: {p}")

    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise DatasetError(f"invalid YAML in {p}: {exc}") from exc

    if not isinstance(raw, list):
        raise DatasetError(f"dataset {p} must be a YAML list of cases")

    try:
        cases = [TestCase.model_validate(item) for item in raw]
    except ValidationError as exc:
        raise DatasetError(f"invalid test case in {p}: {exc}") from exc

    ids = [c.id for c in cases]
    if len(set(ids)) != len(ids):
        raise DatasetError(f"duplicate case ids in {p}")

    return cases
