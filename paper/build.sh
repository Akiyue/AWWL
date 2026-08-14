#!/usr/bin/env bash
# Build the draft, refresh the figures from the latest run, and surface what
# is still open.
#
#   bash paper/build.sh          full build (bibliography included)
#   QUICK=1 bash paper/build.sh  one pass, for checking prose while editing
#
# Figures are copied from runs/ rather than referenced in place: the sweep
# overwrites them, and a paper that silently changes its own figures when an
# unrelated job finishes is not reproducible.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "${HERE}")"
QUICK=${QUICK:-0}

mkdir -p "${HERE}/figures"
# `awwl spectrum` writes beside the sweep it read (runs/phase0/), while
# compare-samples defaults to runs/. Search both rather than silently keeping
# a stale copy when the newest figure lands in the other directory.
for f in spectrum compare curriculum; do
  for src in "${ROOT}/runs/phase0/${f}.png" "${ROOT}/runs/${f}.png"; do
    if [ -f "${src}" ]; then
      cp "${src}" "${HERE}/figures/${f}.png"
      echo "figure: ${f}.png  <- ${src#${ROOT}/}"
      break
    fi
  done
done

cd "${HERE}" || exit 1

# Check the toolchain before invoking it. The build redirects LaTeX's output to
# /dev/null, so a missing binary would otherwise surface as an empty failure
# with no log to read -- which is exactly how this script behaved on a machine
# without TeX installed.
need=$([ "${QUICK}" = "1" ] && echo pdflatex || echo latexmk)
if ! command -v "${need}" >/dev/null 2>&1; then
  echo "no ${need} on this machine." >&2
  echo >&2
  echo "The tables and figures are generated where the data is; the PDF is" >&2
  echo "built where TeX is. They need not be the same machine:" >&2
  echo >&2
  echo "  on the server:  python scripts/make_tables.py" >&2
  echo "                  awwl spectrum --root runs/phase0 ..." >&2
  echo "                  git add -A paper/tables paper/figures && git push" >&2
  echo "  where TeX is:   git pull && bash paper/build.sh" >&2
  echo >&2
  echo "Or install TeX here: apt-get install texlive-latex-recommended \\" >&2
  echo "  texlive-latex-extra texlive-fonts-recommended latexmk" >&2
  exit 2
fi

if [ "${QUICK}" = "1" ]; then
  pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null 2>&1
  status=$?
else
  latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex >/dev/null 2>&1
  status=$?
fi

# Judge the build by the log and the artefact, not by latexmk's exit code
# alone: it returns non-zero for conditions that still produce a correct PDF
# (a wanted rerun, a stale .fdb after files are removed by hand), and a build
# script that cries failure on a good build gets ignored on a bad one.
# `grep -c` already prints 0 when there is no match; the `|| echo 0` this
# used to carry appended a second line and broke the numeric test.
errors=$(grep -c "^!" main.log 2>/dev/null || true)
errors=${errors:-0}

if [ "${errors}" -gt 0 ] || [ ! -f main.pdf ]; then
  echo
  echo "BUILD FAILED — first errors:" >&2
  grep -m 10 -E "^!|^l\.[0-9]+" main.log >&2 || true
  echo
  echo "full log: ${HERE}/main.log" >&2
  exit 1
fi

if [ "${status}" -ne 0 ]; then
  echo "note: latexmk exited ${status} but the log is clean and the PDF is present" >&2
fi

pages=$(grep -oE "Output written on main.pdf \([0-9]+ page" main.log | grep -oE "[0-9]+" | tail -1)
echo
echo "built main.pdf (${pages:-?} pages)"

# Unresolved references are the failure that survives a clean build.
if grep -q "undefined" main.log; then
  echo
  echo "undefined references:" >&2
  grep -E "Reference .* undefined|Citation .* undefined" main.log | sort -u >&2
fi

# Count macro uses, not the rendered words: the uppercase text appears only in
# the \newcommand, so grepping for it found the definition and nothing else.
todo=$(grep -oh '\\todo{' sections/*.tex 2>/dev/null | wc -l)
pending=$(grep -oh '\\pending{' sections/*.tex 2>/dev/null | wc -l)
echo "open items: ${todo} TODO, ${pending} PENDING  (in colour in the PDF)"
if [ "$((todo + pending))" -gt 0 ]; then
  grep -Hn '\\todo{\|\\pending{' sections/*.tex | sed 's/^/  /'
fi
