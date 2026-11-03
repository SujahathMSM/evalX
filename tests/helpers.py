"""Small helpers shared by the test modules."""

import re

# Rich draws table borders with box characters on a UTF-8 terminal and falls
# back to ASCII ("|") when the output encoding can't show them (e.g. a
# Windows console on cp1252), so accept both.
_CELL_BORDER = re.compile(r"[│|┃]")


def table_cells(row: str) -> list[str]:
    """The non-empty cells of one rendered Rich table row."""
    return [c.strip() for c in _CELL_BORDER.split(row) if c.strip()]
