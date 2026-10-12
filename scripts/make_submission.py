"""Build the submission package: one flat LaTeX source, its figures, and the PDF.

Editorial Manager flattens uploaded folders and compiles with pdfLaTeX, so the
manuscript's layout -- section files under ``sections/``, generated tables
under ``tables/``, macros under ``generated/``, figures under ``figures/`` and a
BibTeX database -- does not survive an upload as it stands. This script writes
``paper/submission/package/``:

``manuscript.tex``  every ``\\input`` inlined (recursively), the generated
                    number macros inlined, the bibliography inlined from the
                    last build's ``main.bbl``, figure paths flattened, and
                    full-line comments removed
``*.pdf``           every figure the manuscript includes (vector PDF when the
                    build produced one, else the PNG)

and compiles the package in a scratch copy to check that it builds on its own.
Run ``tectonic -X compile main.tex --keep-intermediates`` in ``paper/`` first so
``main.bbl`` is current.

Usage::

    python scripts/make_submission.py
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
OUT = PAPER / "submission" / "package"


def _matching_brace(text: str, start: int) -> int:
    """Index just past the group that opens at ``text[start] == '{'``."""
    assert text[start] == "{", text[start : start + 20]
    depth = 0
    i = start
    while i < len(text):
        c = text[i]
        if c == "\\":
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    raise ValueError("unbalanced braces")


def _read(rel: str) -> str:
    path = PAPER / rel
    if path.suffix != ".tex":
        path = path.with_suffix(".tex")
    return path.read_text(encoding="utf-8")


def _strip_comment_lines(text: str) -> str:
    """Drop lines whose first non-blank character is ``%`` (internal notes)."""
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("%")) + "\n"


def _expand(text: str, depth: int = 0) -> str:
    if depth > 10:
        raise RecursionError("input nesting too deep")
    # \IfFileExists{path}{yes}{no}: keep the branch the build would take.
    out, i = [], 0
    pat = re.compile(r"\\IfFileExists\{")
    while True:
        m = pat.search(text, i)
        if not m:
            out.append(text[i:])
            break
        out.append(text[i : m.start()])
        a = m.end() - 1
        b = _matching_brace(text, a)
        path = text[a + 1 : b - 1]
        c = _matching_brace(text, b)
        d = _matching_brace(text, c)
        yes, no = text[b + 1 : c - 1], text[c + 1 : d - 1]
        # Inside the figure macro the test is on a macro-built name; leave it.
        if "#" in path or "\\" in path:
            out.append(text[m.start() : d])
        else:
            out.append(yes if (PAPER / path).exists() else no)
        i = d
    text = "".join(out)
    text = re.sub(r"\\InputIfFileExists\{([^}]*)\}\{\}\{\}",
                  lambda m: _strip_comment_lines(_read(m.group(1))), text)

    def inline(m: re.Match) -> str:
        return _expand(_strip_comment_lines(_read(m.group(1))), depth + 1)

    return re.sub(r"\\input\{([^}]*)\}", inline, text)


def build(out: Path) -> Path:
    bbl = PAPER / "main.bbl"
    if not bbl.exists():
        raise SystemExit("paper/main.bbl not found: build with --keep-intermediates first")
    text = _strip_comment_lines((PAPER / "main.tex").read_text(encoding="utf-8"))
    text = _expand(text)
    # Bibliography: the formatted list from the last build, no BibTeX run needed.
    text = re.sub(r"\\bibliographystyle\{[^}]*\}\s*\n", "", text)
    text = text.replace("\\bibliography{refs}", bbl.read_text(encoding="utf-8").strip())
    # Figures sit beside the source.
    text = text.replace("\\graphicspath{{figures/}}", "\\graphicspath{{./}}")
    text = text.replace("\\IfFileExists{figures/", "\\IfFileExists{")
    # Draft-only markers: nothing in the finished text uses them.
    text = re.sub(r"\\newcommand\{\\(?:todo|pending)\}[^\n]*\n", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Rebuild only what this script owns; the cover letter and highlights in
    # the same folder are written separately and must survive a rebuild.
    out.mkdir(parents=True, exist_ok=True)
    keep = {"cover_letter.pdf"}
    for f in out.iterdir():
        if f.name.startswith("manuscript") or (f.suffix == ".pdf" and f.name not in keep):
            f.unlink()
    (out / "manuscript.tex").write_text(text, encoding="utf-8")

    figures = sorted(set(re.findall(r"\\guardedfigure\{[^}]*\}\{([^}]*)\}", text)))
    for name in figures:
        vector = (PAPER / "figures" / name).with_suffix(".pdf")
        src = vector if vector.exists() else PAPER / "figures" / name
        if not src.exists():
            raise SystemExit(f"figure missing: {src}")
        shutil.copy2(src, out / src.name)
    leftovers = re.findall(r"\\input\{|sections/|tables/|generated/", text)
    if leftovers:
        raise SystemExit(f"unexpanded paths remain: {sorted(set(leftovers))}")
    print(f"wrote {out / 'manuscript.tex'} and {len(figures)} figures")
    plain_abstract(out)
    return out


def plain_abstract(out: Path) -> None:
    """The abstract as plain text, macros expanded, for the submission form."""
    numbers = (PAPER / "generated" / "numbers.tex").read_text(encoding="utf-8")
    macros = dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}", numbers))
    src = _strip_comment_lines((PAPER / "sections" / "abstract.tex").read_text(encoding="utf-8"))
    t = re.sub(r"\\([A-Za-z]+)(\{\})?", lambda m: macros.get(m.group(1), m.group(0)), src)
    for a, b in {"Fr\\'echet": "Fréchet", "\\mathrm{MRG}": "MRG", "$": "", "~": " ", "--": "–"}.items():
        t = t.replace(a, b)
    t = re.sub(r"\\emph\{([^}]*)\}", r"\1", t).replace("\\", "")
    t = re.sub(r"(?<=\d)-(?=\d)", "–", " ".join(t.split()))
    (out / "abstract.txt").write_text(t + "\n", encoding="utf-8")
    print(f"abstract.txt: {len(t.split())} words as the submission form counts them")


def zip_source(out: Path) -> None:
    """manuscript.tex and its figures in one archive for upload."""
    import zipfile

    figures = sorted(f.name for f in out.glob("*.pdf") if f.name not in ("manuscript.pdf", "cover_letter.pdf"))
    with zipfile.ZipFile(out / "manuscript_source.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for name in ["manuscript.tex", *figures]:
            z.write(out / name, name)
    print(f"manuscript_source.zip: manuscript.tex and {len(figures)} figures")


def check(out: Path) -> None:
    """Compile the package on its own, in a flat scratch folder."""
    with tempfile.TemporaryDirectory() as tmp:
        for f in out.iterdir():
            shutil.copy2(f, tmp)
        res = subprocess.run(["tectonic", "manuscript.tex", "--keep-logs"], cwd=tmp,
                             capture_output=True, text=True)
        log = (Path(tmp) / "manuscript.log").read_text(errors="ignore") if (Path(tmp) / "manuscript.log").exists() else ""
        pages = re.findall(r"Output written on manuscript\.xdv \((\d+) pages", log)
        undefined = len(re.findall(r"undefined", log))
        if res.returncode != 0 or not pages:
            sys.stderr.write(res.stderr[-2000:])
            raise SystemExit("the flat package does not compile")
        shutil.copy2(Path(tmp) / "manuscript.pdf", out / "manuscript.pdf")
        print(f"flat package compiles: {pages[-1]} pages, {undefined} undefined references")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--no-check", action="store_true")
    args = ap.parse_args()
    out = build(args.out)
    if not args.no_check:
        check(out)
    zip_source(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
