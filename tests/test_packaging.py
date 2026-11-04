"""Release hygiene: the version, changelog and package metadata agree."""

import re
import tomllib
from pathlib import Path

import evalx

ROOT = Path(__file__).resolve().parents[1]


def test_changelog_has_a_section_for_this_version():
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert f"## [{evalx.__version__}]" in changelog


def test_pyproject_reads_version_from_the_package():
    meta = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert meta["project"]["name"] == "evalx-gate"
    assert "version" in meta["project"]["dynamic"]
    assert meta["tool"]["hatch"]["version"]["path"] == "evalx/__init__.py"
    assert meta["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == ["evalx"]


def test_readme_links_work_on_pypi():
    """PyPI renders the README without the repo, so no relative links/images."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    targets = re.findall(r"\]\(([^)]+)\)", readme) + re.findall(r'src="([^"]+)"', readme)
    relative = [t for t in targets if not re.match(r"(https?://|#|mailto:)", t)]
    assert relative == []
