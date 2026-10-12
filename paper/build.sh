#!/usr/bin/env bash
# Build the manuscript and surface what is still open.
#
#   bash paper/build.sh                 static checks, then a full build
#   QUICK=1 bash paper/build.sh         one pdflatex pass (latexmk route only)
#   FIGURES_ONLY=1 bash paper/build.sh  check the figures, skip the build
#   SKIP_CHECKS=1 bash paper/build.sh   skip tests/test_manuscript.py
#   PYTHON=/path/to/python              interpreter for the static checks
#
# Figures. The plotting commands are run with --out paper/figures, and this
# script never copies, regenerates or overwrites a figure. An earlier version
# copied runs/phase0/*.png over paper/figures on every run, which replaced the
# current spectrum.png with a stale copy in an old style. What remains is a
# warning: a file of the same name under runs/ that is newer than the paper
# copy means a plotting command wrote to its old default directory, and the
# paper copy may be the stale one. Regenerate into paper/figures if so.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "${HERE}")"
QUICK=${QUICK:-0}
FIGURES_ONLY=${FIGURES_ONLY:-0}
SKIP_CHECKS=${SKIP_CHECKS:-0}
PYTHON=${PYTHON:-python}

# --- figures: report only, never copy ----------------------------------------
newer=0
for dst in "${HERE}"/figures/*.png "${HERE}"/figures/*.pdf; do
  [ -f "${dst}" ] || continue
  name="$(basename "${dst}")"
  for src in "${ROOT}/runs/phase0/${name}" "${ROOT}/runs/figures/${name}" "${ROOT}/runs/${name}"; do
    if [ -f "${src}" ] && [ "${src}" -nt "${dst}" ]; then
      echo "warning: ${src#${ROOT}/} is newer than paper/figures/${name}; not copied." >&2
      newer=$((newer + 1))
    fi
  done
done
if [ "${newer}" -gt 0 ]; then
  echo "         Check which copy is current, and regenerate into paper/figures" >&2
  echo "         (the plotting commands take --out) rather than copying by hand." >&2
fi

if [ "${FIGURES_ONLY}" = "1" ]; then
  echo "figures checked (${newer} newer copies under runs/); skipping the build"
  exit 0
fi

cd "${HERE}" || exit 1

# --- static checks -------------------------------------------------------------
# Dangling \ref and \cite, missing \input targets and unbuilt figures, in
# milliseconds, before a build that reports them only at the end.
if [ "${SKIP_CHECKS}" != "1" ]; then
  if [ -f "${ROOT}/tests/test_manuscript.py" ] \
     && "${PYTHON}" -c "import pytest" >/dev/null 2>&1; then
    if ! (cd "${ROOT}" && "${PYTHON}" -m pytest -q tests/test_manuscript.py); then
      echo "manuscript checks failed (SKIP_CHECKS=1 to build anyway)" >&2
      exit 1
    fi
  else
    echo "note: static checks skipped (no tests/test_manuscript.py or no pytest for ${PYTHON})" >&2
  fi
fi

# A stamp taken before the build: a main.pdf older than it is a leftover from
# an earlier build, not evidence that this one succeeded.
stamp="$(mktemp)"
trap 'rm -f "${stamp}"' EXIT

report() {
  # $1: engine name, for the summary line.
  pages=$(grep -oE "Output written on main\.(pdf|xdv) \([0-9]+ page" main.log 2>/dev/null \
          | grep -oE "[0-9]+" | tail -1)
  echo
  echo "built main.pdf (${pages:-?} pages, $1)"
  overfull=$(grep -c "^Overfull" main.log 2>/dev/null || true)
  echo "overfull boxes: ${overfull:-0}"
  # Unresolved references are the failure that survives a clean build.
  if grep -qE "Reference .* undefined|Citation .* undefined" main.log 2>/dev/null; then
    echo
    echo "undefined references:" >&2
    grep -E "Reference .* undefined|Citation .* undefined" main.log | sort -u >&2
  fi
  # A label defined twice resolves silently to the last definition.
  if grep -qE "Label .* multiply defined" main.log 2>/dev/null; then
    echo
    echo "multiply defined labels:" >&2
    grep -E "Label .* multiply defined" main.log | sort -u >&2
  fi
  # Count macro uses, not the rendered words.
  todo=$(grep -oh '\\todo{' sections/*.tex 2>/dev/null | wc -l)
  pending=$(grep -oh '\\pending{' sections/*.tex 2>/dev/null | wc -l)
  echo "open items: ${todo} TODO, ${pending} PENDING  (in colour in the PDF)"
  if [ "$((todo + pending))" -gt 0 ]; then
    grep -Hn '\\todo{\|\\pending{' sections/*.tex | sed 's/^/  /'
  fi
}

# --- tectonic ------------------------------------------------------------------
# Preferred when present: a single binary that needs no root and fetches the
# packages it needs on demand. The first build on a cold cache downloads the
# bundle and can take ten minutes; later builds take seconds.
#
#   curl -fsSL https://github.com/tectonic-typesetting/tectonic/releases\
# /download/tectonic%400.15.0/tectonic-0.15.0-x86_64-unknown-linux-musl.tar.gz \
#     | tar xz -C ~/.local/bin
#
# Tectonic 0.15 always reports "internal consistency problem when checking if
# main.bbl changed" and stops at six passes, even for a one-citation article;
# the references are resolved by then, so the message is not a build failure.
if command -v tectonic >/dev/null 2>&1; then
  tectonic -X compile main.tex --outdir . --keep-logs --keep-intermediates \
      >/dev/null 2>&1
  status=$?
  if [ "${status}" -ne 0 ] || [ ! main.pdf -nt "${stamp}" ]; then
    echo "BUILD FAILED (tectonic) -- first errors:" >&2
    grep -m 10 -E "^!|^error|^l\.[0-9]+" main.log 2>/dev/null >&2 || true
    echo "full log: ${HERE}/main.log" >&2
    exit 1
  fi
  report tectonic
  exit 0
fi

# --- TeX Live ------------------------------------------------------------------
need=$([ "${QUICK}" = "1" ] && echo pdflatex || echo latexmk)
if ! command -v "${need}" >/dev/null 2>&1; then
  echo "no ${need} on this machine." >&2
  echo >&2
  echo "The tables and figures are generated where the data is; the PDF is" >&2
  echo "built where TeX is. They need not be the same machine:" >&2
  echo >&2
  echo "  on the server:  python scripts/make_tables.py" >&2
  echo "                  (plotting commands with --out paper/figures)" >&2
  echo "                  git add -A paper/tables paper/figures && git push" >&2
  echo "  where TeX is:   git pull && bash paper/build.sh" >&2
  echo >&2
  echo "Or install TeX here: apt-get install texlive-latex-recommended \\" >&2
  echo "  texlive-latex-extra texlive-fonts-recommended latexmk" >&2
  echo >&2
  echo "Without root, tectonic is a single binary and this script prefers it:" >&2
  echo "  curl -fsSL https://github.com/tectonic-typesetting/tectonic/releases\\" >&2
  echo "/download/tectonic%400.15.0/tectonic-0.15.0-x86_64-unknown-linux-musl.tar.gz \\" >&2
  echo "    | tar xz -C ~/.local/bin" >&2
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
# (a wanted rerun, a stale .fdb after files are removed by hand).
errors=$(grep -c "^!" main.log 2>/dev/null || true)
errors=${errors:-0}

if [ "${errors}" -gt 0 ] || [ ! main.pdf -nt "${stamp}" ]; then
  echo
  echo "BUILD FAILED -- first errors:" >&2
  grep -m 10 -E "^!|^l\.[0-9]+" main.log >&2 || true
  echo
  echo "full log: ${HERE}/main.log" >&2
  exit 1
fi

if [ "${status}" -ne 0 ]; then
  echo "note: ${need} exited ${status} but the log is clean and the PDF is present" >&2
fi

report "${need}"
