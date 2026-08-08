"""Tests for the statutory numbering machine.

These are regression tests for bugs that were actually hit while building the
corpus, not hypotheticals. Each one corresponds to a specific way 8 CFR 214.2
broke the chunker.

    python -m pytest tests/ -v      (or: python tests/test_chunk.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foga.chunk import NumberingStack, _segment_line, _statutory_blocks  # noqa: E402


def _path(tokens: list[str]) -> str:
    s = NumberingStack()
    for t in tokens:
        s.push_marker(t)
    return s.path


def test_basic_nesting():
    assert _path(["a"]) == "(a)"
    assert _path(["a", "1"]) == "(a)(1)"
    assert _path(["a", "1", "2"]) == "(a)(2)"
    assert _path(["a", "1", "i", "ii"]) == "(a)(1)(ii)"


def test_pops_to_shallower_level():
    # (a)(1)(i) followed by (2) means the numeric level advanced, not a new one.
    assert _path(["a", "1", "i", "2"]) == "(a)(2)"
    assert _path(["a", "1", "i", "b"]) == "(b)"


def test_roman_vs_alpha_ambiguity():
    """The bug that mattered most.

    '(i)' is both the 9th letter and the 1st roman numeral. In 8 CFR 214.2,
    (h) is H-1B and (i) is media representatives. When the parser sat at (h)(4)
    and met '(i)', the naive rule read it as (h)'s successor and moved 166
    H-visa chunks under the media subsection.
    """
    # Deeper level wins when we are already inside a subsection.
    assert _path(["h", "4", "i"]) == "(h)(4)(i)"
    assert _path(["h", "4", "i", "ii", "B"]) == "(h)(4)(ii)(B)"
    # But at the top level, (h) -> (i) is a genuine letter advance.
    assert _path(["g", "h", "i", "j"]) == "(j)"


def test_same_kind_may_repeat_at_depth():
    # (a)(1)(i)(A)(1)(i) is normal CFR numbering: num and roman each recur.
    assert _path(["a", "1", "i", "A", "1", "i"]) == "(a)(1)(i)(A)(1)(i)"


def test_restart_after_table_of_contents():
    """The section-contents table lists (a)..(w); the body then restarts at (a).

    A kind is never nested directly inside itself, so this must replace the
    level rather than produce (w)(a).
    """
    assert _path(["w", "a"]) == "(a)"
    assert _path(["w", "a", "b"]) == "(b)"


def test_out_of_sequence_marker():
    # [Reserved] paragraphs and tables cause gaps; do not crash or nest wrongly.
    assert _path(["a", "c"]) == "(c)"
    assert _path(["a", "1", "5"]) == "(a)(5)"


def test_inline_em_dash_markers():
    """The CFR opens child subsections mid-line after an em dash."""
    line = "(a) Foreign government officials—(1) General. The determination by a consular officer "
    segs = _segment_line(line)
    assert [m for m, _ in segs] == ["a", "1"]


def test_cross_references_are_not_structure():
    """'section 101(a)(15)(F)' is a citation, not a subsection opening."""
    line = "The alien must qualify under section 101(a)(15)(F) of the Act (see paragraph (b)(2))."
    segs = _segment_line(line)
    assert [m for m, _ in segs] == [None]


def test_blocks_carry_paths():
    text = (
        "(f) Students in colleges—(1) Admission of student.\n"
        "The student must present a Form I-20.\n"
        "(9) Off-campus employment.\n"
        "(i) On-campus employment. On-campus employment must be performed on premises.\n"
    )
    blocks = _statutory_blocks(text)
    paths = [b.path for b in blocks]
    assert "(f)(1)" in paths
    assert "(f)(9)" in paths
    assert "(f)(9)(i)" in paths


if __name__ == "__main__":
    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {name}")
                passed += 1
            except AssertionError as exc:
                print(f"  FAIL  {name}: {exc}")
                failed += 1
    print(f"\n{passed} passed, {failed} failed")
    raise SystemExit(1 if failed else 0)
