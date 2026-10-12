"""Static checks on the manuscript, for a machine with no TeX.

The paper is written on the training server and typeset elsewhere, so the
failures that a build would catch -- a \\ref with no \\label, a \\cite with no
bib entry, an \\input of a file that was never generated -- can sit in the
draft for weeks. These reproduce the parts of that check that do not need a
compiler, and they run in the ordinary suite.

They deliberately do not check LaTeX syntax: that is the compiler's job and a
regex approximation of it would fail on valid input.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PAPER = Path(__file__).resolve().parents[1] / "paper"
SECTIONS = sorted((PAPER / "sections").glob("*.tex")) if (PAPER / "sections").is_dir() else []

pytestmark = pytest.mark.skipif(not SECTIONS, reason="no manuscript in this checkout")


def _body() -> str:
    """Every source the build reads, concatenated."""
    parts = [(PAPER / "main.tex").read_text(encoding="utf-8")]
    parts += [f.read_text(encoding="utf-8") for f in SECTIONS]
    parts += [f.read_text(encoding="utf-8") for f in sorted((PAPER / "tables").glob("*.tex"))]
    return "\n".join(parts)


def _strip_comments(text: str) -> str:
    # A comment can carry an example \ref that has no label, which is fine.
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", line) for line in text.splitlines())


def _braced_args(text: str, macro: str, count: int) -> list[list[str]]:
    """Every call of ``macro``, as its ``count`` brace-delimited arguments.

    Written by hand rather than with a regex because a caption argument
    contains nested braces, and a non-greedy match stops at the first inner
    one -- which is how ``\\guardedfigure``'s label argument came to look
    absent to a first version of this check.
    """
    out: list[list[str]] = []
    for match in re.finditer(re.escape("\\" + macro) + r"(?=\{)", text):
        i, args = match.end(), []
        for _ in range(count):
            if i >= len(text) or text[i] != "{":
                break
            depth, start = 0, i + 1
            while i < len(text):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                i += 1
            args.append(text[start:i])
            i += 1
        if len(args) == count:
            out.append(args)
    return out


def _labels(body: str) -> set[str]:
    """Labels however they are declared: directly, or via \\guardedfigure."""
    labels = set(re.findall(r"\\label\{([^}]+)\}", body))
    labels.update(args[3] for args in _braced_args(body, "guardedfigure", 4))
    return labels


def test_every_ref_has_a_label():
    body = _strip_comments(_body())
    labels = _labels(body)
    refs = set(re.findall(r"\\(?:ref|autoref|eqref)\{([^}]+)\}", body))
    missing = sorted(refs - labels)
    assert not missing, f"referenced but never labelled: {missing}"


def test_every_citation_is_in_the_bibliography():
    body = _strip_comments(_body())
    bib = (PAPER / "refs.bib").read_text(encoding="utf-8")
    keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    cited: set[str] = set()
    for group in re.findall(r"\\cite[tp]?\*?(?:\[[^\]]*\])*\{([^}]+)\}", body):
        cited.update(k.strip() for k in group.split(","))
    missing = sorted(cited - keys)
    assert not missing, f"cited but not in refs.bib: {missing}"


def test_every_input_resolves():
    main = _strip_comments((PAPER / "main.tex").read_text(encoding="utf-8"))
    for name in re.findall(r"\\input\{([^}]+)\}", main):
        target = PAPER / (name if name.endswith(".tex") else name + ".tex")
        assert target.exists(), f"\\input{{{name}}} has no file at {target}"


# Section files whose content was merged into another section's file. Their
# \input was removed from main.tex and the files were then deleted; a copy
# restored from an old checkout, or an \input of one, would print the
# material twice or an empty section.
MERGED_SECTIONS = ("family", "replication", "extensions", "limitations")


def _section_inputs() -> list[str]:
    """Every ``\\input{sections/...}`` target, one entry per call, comments removed."""
    names: list[str] = []
    for src in [PAPER / "main.tex", *SECTIONS]:
        text = _strip_comments(src.read_text(encoding="utf-8"))
        names += re.findall(r"\\input\{sections/([^}]+?)(?:\.tex)?\}", text)
    return names


def test_merged_sections_are_not_input():
    inputs = _section_inputs()
    assert inputs, "no \\input{sections/...} found; the parser has drifted"
    stale = sorted(set(inputs) & set(MERGED_SECTIONS))
    assert not stale, f"merged sections still input: {stale}"
    present = sorted(n for n in MERGED_SECTIONS if (PAPER / "sections" / f"{n}.tex").exists())
    assert not present, f"merged section files are back in paper/sections: {present}"


def _table_inputs() -> list[str]:
    """Every ``\\input{tables/...}`` target, one entry per call, comments removed."""
    names: list[str] = []
    for src in [PAPER / "main.tex", *SECTIONS]:
        text = _strip_comments(src.read_text(encoding="utf-8"))
        names += re.findall(r"\\input\{tables/([^}]+?)(?:\.tex)?\}", text)
    return names


def test_no_generated_table_is_input_twice():
    """A table input twice prints twice and defines its label twice.

    Tables move between the main text and the appendices from one revision to
    the next (the replication details, the confound tables), and a move that
    adds the new \\input without removing the old one still builds.
    """
    inputs = _table_inputs()
    assert inputs, "no \\input{tables/...} found; the parser has drifted"
    twice = sorted({name for name in inputs if inputs.count(name) > 1})
    assert not twice, f"tables input more than once: {twice}"


def test_every_supplementary_file_is_input_exactly_once():
    """Each sections/supp_*.tex is input once, under Appendix D.

    The supplementary files carry the floats moved out of the main text, with
    their labels. One that is never input leaves every reference to its floats
    dangling; one input twice defines each of its labels twice.
    """
    inputs = _section_inputs()
    supp = sorted(f.stem for f in SECTIONS if f.stem.startswith("supp_"))
    wrong = {name: inputs.count(name) for name in supp if inputs.count(name) != 1}
    assert not wrong, f"supplementary files not input exactly once: {wrong}"


def test_guarded_inputs_and_figures_are_present():
    """Every guarded \\input and \\includegraphics has its artefact built.

    These degrade to a placeholder box rather than failing the build, which is
    what makes them worth a test: a missing figure is invisible in the build
    log and shows up only as a grey rectangle in the PDF nobody reads closely.
    """
    body = _strip_comments("\n".join(f.read_text(encoding="utf-8") for f in SECTIONS))
    missing: list[str] = []
    for name in re.findall(r"\\guardedfigure\{[^}]*\}\{([^}]+)\}", body):
        if not (PAPER / "figures" / name).exists():
            missing.append(f"figures/{name}")
    for name in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", body):
        if not (PAPER / "figures" / name).exists():
            missing.append(f"figures/{name}")
    for name in re.findall(r"\\IfFileExists\{(tables/[^}]+)\}", body):
        if not (PAPER / name).exists():
            missing.append(name)
    assert not missing, f"referenced but not built: {sorted(set(missing))}"


def test_generated_tables_are_not_edited_by_hand():
    """The tables carry a do-not-edit banner; check it is still there.

    A table edited in place is the one way a hand-typed number can re-enter a
    paper that claims none of them are transcribed.
    """
    for table in sorted((PAPER / "tables").glob("*.tex")):
        head = table.read_text(encoding="utf-8")[:200]
        # Any generator, not one named script: the band table is recomputed
        # from the sampled images rather than read from the ledger, so it has
        # its own (slower) generator and its own banner.
        assert "Generated by scripts/" in head, f"{table.name} lost its banner"


NUMBERS = PAPER / "generated" / "numbers.tex"

# Hand-set numbers that main.tex carried before generated/numbers.tex existed.
# They were retired once every section read the generated macro of the same
# meaning; each is listed so that none can come back under its old name.
LEGACY_MACROS = (
    "deficitHigh",
    "correction",
    "boostValue",
    "boostCI",
    "shortfall",
    "fidMSE",
    "fidAWWL",
    "fidMSEboost",
    "fidAWWLboost",
    "ddpmDelta",
    "ddpmP",
    "dbDeficitMSE",
    "dbDeficitAWWL",
    "dbCorrection",
    "dbResidual",
)


def _macro_values(path: Path) -> dict[str, str]:
    """``\\newcommand{\\name}{value}`` definitions, values with nested braces."""
    text = path.read_text(encoding="utf-8")
    out: dict[str, str] = {}
    for match in re.finditer(r"\\newcommand\{\\(\w+)\}\{", text):
        i, depth = match.end(), 1
        while depth and i < len(text):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        out[match.group(1)] = text[match.end() : i - 1]
    return out


def _uses(text: str, name: str) -> bool:
    # \fidMSE must not match \fidMSEmean: a control word ends at a non-letter.
    return re.search(re.escape("\\" + name) + r"(?![A-Za-z])", text) is not None


@pytest.mark.skipif(not NUMBERS.exists(), reason="generated/numbers.tex not built")
def test_no_placeholder_number_is_printed():
    """No section prints a macro whose value is still the ``\\textbf{??}`` mark.

    make_numbers.py writes that mark for a quantity whose ledger rows do not
    exist yet. It compiles without complaint, so only a check like this one
    keeps it out of a submitted PDF.
    """
    placeholders = [n for n, v in _macro_values(NUMBERS).items() if "??" in v]
    body = _strip_comments("\n".join(f.read_text(encoding="utf-8") for f in SECTIONS))
    used = sorted(n for n in placeholders if _uses(body, n))
    assert not used, f"placeholder numbers printed in the manuscript: {used}"


def test_no_legacy_macro_remains():
    """Every number comes from generated/numbers.tex, none from main.tex.

    main.tex once defined measured numbers by hand, and a re-run of the
    analysis did not update them. None may be defined or used again, and
    main.tex may not set any other number by hand.
    """
    main = (PAPER / "main.tex").read_text(encoding="utf-8")
    assert "LEGACY" not in main, "main.tex still carries a LEGACY block"
    # \renewcommand of LaTeX's float parameters is layout, not a measurement.
    by_hand = re.findall(r"\\newcommand\{\\(\w+)\}\{[-+]?[\d.]+\}", main)
    assert not by_hand, f"numbers set by hand in main.tex: {by_hand}"
    sources = [PAPER / "main.tex", *SECTIONS, *sorted((PAPER / "tables").glob("*.tex"))]
    stale = sorted(
        f"{src.name}: \\{name}"
        for src in sources
        for name in LEGACY_MACROS
        if _uses(_strip_comments(src.read_text(encoding="utf-8")), name)
    )
    assert not stale, f"legacy macros still used: {stale}"


def test_every_guarded_figure_file_exists():
    """Each \\guardedfigure names a file that is present in paper/figures.

    Parsed with the brace matcher, not a regex, so a caption with nested
    braces cannot hide a call.
    """
    body = _strip_comments("\n".join(f.read_text(encoding="utf-8") for f in SECTIONS))
    calls = _braced_args(body, "guardedfigure", 4)
    assert calls, "no \\guardedfigure call found; the parser has drifted"
    missing = sorted({args[1] for args in calls if not (PAPER / "figures" / args[1]).is_file()})
    assert not missing, f"guarded figures with no file: {missing}"


def _drop_arg(text: str, macro: str, count: int, which: int) -> str:
    """``text`` with argument ``which`` (0-based) of every ``macro`` call removed."""
    out, pos = [], 0
    for match in re.finditer(re.escape("\\" + macro) + r"(?=\{)", text):
        if match.start() < pos:
            continue
        i, spans = match.end(), []
        for _ in range(count):
            if i >= len(text) or text[i] != "{":
                break
            depth, start = 0, i
            while i < len(text):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                i += 1
            spans.append((start, i + 1))
            i += 1
        if len(spans) == count:
            start, end = spans[which]
            out.append(text[pos:start])
            pos = end
    out.append(text[pos:])
    return "".join(out)


def test_every_float_is_cited_in_the_text():
    """Every figure, table and algorithm is referred to from the prose.

    Elsevier asks for every display item to be cited in the text; a float
    cited only from another float's caption does not count, so captions are
    removed before the citations are collected.
    """
    prose = _strip_comments(
        "\n".join(
            [(PAPER / "main.tex").read_text(encoding="utf-8")]
            + [f.read_text(encoding="utf-8") for f in SECTIONS]
        )
    )
    # Only the tables the manuscript inputs: a generator may still write a
    # table the text no longer prints (the per-metric comparison tables that
    # Table compare-other replaced), and its label is not a float of the paper.
    printed = [PAPER / "tables" / f"{name}.tex" for name in sorted(set(_table_inputs()))]
    tables = "\n".join(t.read_text(encoding="utf-8") for t in printed if t.exists())
    tables = _strip_comments(tables)
    floats = set(re.findall(r"\\label\{((?:fig|tab|alg):[^}]+)\}", prose + "\n" + tables))
    floats.update(args[3] for args in _braced_args(prose, "guardedfigure", 4))
    running = _drop_arg(_drop_arg(prose, "guardedfigure", 4, 2), "caption", 1, 0)
    cited = set(re.findall(r"\\(?:ref|autoref)\{([^}]+)\}", running))
    uncited = sorted(floats - cited)
    assert not uncited, f"floats never cited in the text: {uncited}"


ABSTRACT_WORD_LIMIT = 250  # Neurocomputing's Guide for Authors


def _abstract_words() -> list[str]:
    """The abstract as printed: macros replaced by their generated values."""
    text = _strip_comments((PAPER / "sections" / "abstract.tex").read_text(encoding="utf-8"))
    values = _macro_values(NUMBERS)
    for _ in range(3):  # a value may itself contain a macro
        text = re.sub(r"\\(\w+)(?:\{\})?", lambda m: values.get(m.group(1), m.group(0)), text)
    text = re.sub(r"\\(?:emph|textit|textbf|mathrm|text|num)\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\SI\{([^}]*)\}\{[^}]*\}", r"\1", text)
    text = re.sub(r"\\[,;!]|[~$]", " ", text)
    # A control word left at this point separates words: $64\times64$ prints
    # as two numbers, not one.
    text = re.sub(r"\\[A-Za-z]+", " ", text)
    text = re.sub(r"[{}]", "", text)
    return [w for w in text.split() if re.search(r"[A-Za-z0-9]", w)]


@pytest.mark.skipif(not NUMBERS.exists(), reason="generated/numbers.tex not built")
def test_abstract_is_within_the_word_limit():
    words = _abstract_words()
    assert len(words) <= ABSTRACT_WORD_LIMIT, (
        f"abstract is {len(words)} words after macro expansion (limit {ABSTRACT_WORD_LIMIT})"
    )


@pytest.mark.skipif(not NUMBERS.exists(), reason="generated/numbers.tex not built")
def test_bold_p_values_are_significant():
    """A hand-built table that bolds a corrected p-value must bold only p < 0.05.

    The generated tables decide bolding from the value; a table assembled in a
    section file from macros hard-codes \\textbf{}, and a re-run that moves a
    p-value across 0.05 would leave a bold mark on a non-significant result.
    """
    values = _macro_values(NUMBERS)
    bad = []
    for tex in SECTIONS:
        text = tex.read_text(encoding="utf-8")
        for name in re.findall(r"\\textbf\{\\([A-Za-z]+(?:Holm|P))\}", text):
            raw = values.get(name, "")
            if "<10^" in raw:
                continue
            try:
                if float(raw) >= 0.05:
                    bad.append(f"{tex.name}: \\{name} = {raw}")
            except ValueError:
                bad.append(f"{tex.name}: \\{name} has no numeric value ({raw!r})")
    assert not bad, "bold p-values that are not below 0.05:\n" + "\n".join(bad)


def test_plain_abstract_fits_the_submission_form():
    """The abstract as the submission system counts it: whitespace-separated tokens.

    Editorial Manager counts the pasted plain text, in which a sign, an
    equals sign or a bracketed interval is a token of its own; a count that
    joins them can pass while the form rejects the abstract. The plain text
    is written to paper/submission/package/abstract.txt by the packaging step.
    """
    plain = PAPER / "submission" / "package" / "abstract.txt"
    if not plain.exists():
        pytest.skip("submission package not built")
    tokens = plain.read_text(encoding="utf-8").split()
    assert len(tokens) <= ABSTRACT_WORD_LIMIT, f"plain abstract is {len(tokens)} tokens"
