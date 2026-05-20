#!/usr/bin/env bash
# After release_pip.sh has published a new version to PyPI, this script:
#   1. Waits for the new pywapor sdist to appear on PyPI
#   2. Reads its sha256 from the PyPI JSON API
#   3. Updates recipe/meta.yaml in the feedstock (version, sha256, build number)
#   4. Pushes a release branch to the fork and opens a PR against conda-forge
#
# Usage: ./release_conda.sh
# Env overrides:
#   FEEDSTOCK_DIR   path to local feedstock clone   (default: ../pywapor-feedstock)
#   UPSTREAM_REPO   conda-forge feedstock slug      (default: conda-forge/pywapor-feedstock)
#   WAIT_TIMEOUT    max seconds to wait on PyPI     (default: 1200)
#   WAIT_INTERVAL   poll interval in seconds        (default: 15)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

FEEDSTOCK_DIR="${FEEDSTOCK_DIR:-${SCRIPT_DIR}/../pywapor-feedstock}"
UPSTREAM_REPO="${UPSTREAM_REPO:-conda-forge/pywapor-feedstock}"
PYPI_NAME="pywapor"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-1200}"
WAIT_INTERVAL="${WAIT_INTERVAL:-15}"
PYPROJECT="$SCRIPT_DIR/pyproject.toml"

if [[ ! -f "$PYPROJECT" ]]; then
  echo "error: $PYPROJECT not found" >&2
  exit 1
fi
if [[ ! -d "$FEEDSTOCK_DIR/recipe" ]]; then
  echo "error: feedstock not found at $FEEDSTOCK_DIR (set FEEDSTOCK_DIR=...)" >&2
  exit 1
fi
command -v gh >/dev/null || { echo "error: gh CLI not installed" >&2; exit 1; }
command -v python3 >/dev/null || { echo "error: python3 not installed" >&2; exit 1; }

# Read version from pyproject.toml (same parser as release_pip.sh).
VERSION=$(awk '
  /^\[project\]/ { in_project = 1; next }
  /^\[/          { in_project = 0 }
  in_project && /^version[[:space:]]*=/ {
    match($0, /"[^"]+"/)
    print substr($0, RSTART + 1, RLENGTH - 2)
    exit
  }
' "$PYPROJECT")

if [[ -z "$VERSION" ]]; then
  echo "error: could not read version from $PYPROJECT" >&2
  exit 1
fi
echo "Releasing ${PYPI_NAME} ${VERSION} to conda-forge"

# Poll PyPI for the sdist and grab its sha256.
URL="https://pypi.org/pypi/${PYPI_NAME}/${VERSION}/json"
echo "Waiting for ${PYPI_NAME}==${VERSION} sdist on PyPI..."
START=$(date +%s)
SHA256=""
while :; do
  SHA256=$(python3 - "$URL" <<'PY' 2>/dev/null || true
import json, sys, urllib.request
try:
    data = json.load(urllib.request.urlopen(sys.argv[1], timeout=10))
except Exception:
    sys.exit(0)
for u in data.get("urls", []):
    if u.get("packagetype") == "sdist":
        print(u["digests"]["sha256"])
        break
PY
  )
  if [[ -n "$SHA256" ]]; then
    break
  fi
  if (( $(date +%s) - START >= WAIT_TIMEOUT )); then
    echo
    echo "error: timed out after ${WAIT_TIMEOUT}s waiting for PyPI" >&2
    exit 1
  fi
  printf '.'
  sleep "$WAIT_INTERVAL"
done
echo
echo "sha256: $SHA256"

cd "$FEEDSTOCK_DIR"

if ! git diff-index --quiet HEAD --; then
  echo "error: feedstock working tree has uncommitted changes" >&2
  git status --short >&2
  exit 1
fi

# Ensure we have an 'upstream' remote pointing at conda-forge.
if ! git remote get-url upstream >/dev/null 2>&1; then
  echo "Adding upstream remote -> https://github.com/${UPSTREAM_REPO}.git"
  git remote add upstream "https://github.com/${UPSTREAM_REPO}.git"
fi
EXPECTED="https://github.com/${UPSTREAM_REPO}.git"
ACTUAL=$(git remote get-url upstream)
if [[ "$ACTUAL" != "$EXPECTED" && "$ACTUAL" != "${EXPECTED%.git}" ]]; then
  echo "error: 'upstream' remote is $ACTUAL, expected $EXPECTED" >&2
  exit 1
fi

git fetch --quiet upstream main

BRANCH="release/v${VERSION}"
if git rev-parse --verify "$BRANCH" >/dev/null 2>&1; then
  echo "error: branch $BRANCH already exists locally; delete it and retry" >&2
  exit 1
fi

# Branch off conda-forge/main so the PR is a clean single-commit diff.
git checkout -B "$BRANCH" upstream/main

python3 - "$FEEDSTOCK_DIR/recipe/meta.yaml" "$VERSION" "$SHA256" <<'PY'
import re, sys
path, version, sha = sys.argv[1], sys.argv[2], sys.argv[3]
src = open(path).read()
src, n_ver = re.subn(r'(\{% set version = ")[^"]+(" %\})', rf'\g<1>{version}\g<2>', src, count=1)
src, n_sha = re.subn(r'(sha256:\s*)\S+', rf'\g<1>{sha}', src, count=1)
src, n_num = re.subn(r'(\n  number:\s*)\d+', r'\g<1>0', src, count=1)
if n_ver != 1 or n_sha != 1 or n_num != 1:
    sys.exit(f"failed to rewrite meta.yaml (ver={n_ver} sha={n_sha} num={n_num})")
open(path, "w").write(src)
PY

git add recipe/meta.yaml
git commit -m "bump pywapor to ${VERSION}"
git push -u origin "$BRANCH"

FORK_OWNER=$(git remote get-url origin | sed -E 's#.*github\.com[/:]([^/]+)/.*#\1#')

PR_URL=$(gh pr create \
  --repo "$UPSTREAM_REPO" \
  --base main \
  --head "${FORK_OWNER}:${BRANCH}" \
  --title "bump pywapor to ${VERSION}" \
  --body "Bumps pywapor to v${VERSION} and updates source sha256.")

echo "Opened PR: $PR_URL"
echo "Run ./merge_conda.sh once CI passes (or to watch+merge automatically)."
