#!/usr/bin/env bash
#
# Regenerate the pywapor HTML documentation.
#
# Usage:
#   ./build_docs.sh           # install doc deps + build
#   ./build_docs.sh --clean   # also wipe docs/_build first
#   ./build_docs.sh --no-deps # skip the pip install step
#
set -euo pipefail

# Resolve the docs directory relative to this script, so it works regardless
# of where the repo is checked out or which directory you run it from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCS_DIR="${SCRIPT_DIR}/docs"

INSTALL_DEPS=1
CLEAN=0
for arg in "$@"; do
    case "${arg}" in
        --no-deps) INSTALL_DEPS=0 ;;
        --clean)   CLEAN=1 ;;
        -h|--help)
            # Print the leading comment block (stop at the first blank line).
            sed -n '2,/^$/ s/^# \{0,1\}//p' "${BASH_SOURCE[0]}"
            exit 0
            ;;
        *)
            echo "Unknown option: ${arg}" >&2
            exit 1
            ;;
    esac
done

if [[ "${INSTALL_DEPS}" -eq 1 ]]; then
    echo "==> Installing documentation dependencies"
    # `sphinx-mdinclude` is a maintained drop-in replacement for the
    # unmaintained `m2r2`, providing the `.. mdinclude::` directive used in the
    # .rst files while supporting modern Sphinx/docutils.
    # `lxml[html_clean]` is required because newer lxml moved
    # `lxml.html.clean` into a separate package that nbsphinx still imports.
    pip install "sphinx>=8,<9" nbsphinx "sphinx-rtd-theme>=3" \
        sphinxcontrib-bibtex sphinx-mdinclude sphinx-copybutton "lxml[html_clean]"
fi

cd "${DOCS_DIR}"

if [[ "${CLEAN}" -eq 1 ]]; then
    echo "==> Cleaning previous build"
    make clean
fi

echo "==> Building HTML docs"
make html

echo "==> Done. Open: ${DOCS_DIR}/_build/html/index.html"
