#!/usr/bin/env bash
# Watch CI on the open conda-forge feedstock PR for the current pywapor version,
# then merge it once checks pass.
#
# Usage:
#   ./merge_conda.sh              # find PR by current version in pyproject.toml
#   ./merge_conda.sh <pr-number>  # target a specific PR
#   ./merge_conda.sh <pr-url>
#
# Env overrides:
#   UPSTREAM_REPO   conda-forge feedstock slug   (default: conda-forge/pywapor-feedstock)
#   FEEDSTOCK_DIR   path to local feedstock      (default: ../pywapor-feedstock; used to infer fork owner)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

UPSTREAM_REPO="${UPSTREAM_REPO:-conda-forge/pywapor-feedstock}"
FEEDSTOCK_DIR="${FEEDSTOCK_DIR:-${SCRIPT_DIR}/../pywapor-feedstock}"
PYPROJECT="$SCRIPT_DIR/pyproject.toml"

command -v gh >/dev/null || { echo "error: gh CLI not installed" >&2; exit 1; }

PR_REF="${1:-}"

if [[ -z "$PR_REF" ]]; then
  if [[ ! -f "$PYPROJECT" ]]; then
    echo "error: no PR ref given and $PYPROJECT not found" >&2
    exit 1
  fi
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

  if [[ ! -d "$FEEDSTOCK_DIR/.git" ]]; then
    echo "error: feedstock not found at $FEEDSTOCK_DIR (set FEEDSTOCK_DIR=...)" >&2
    exit 1
  fi
  FORK_OWNER=$(git -C "$FEEDSTOCK_DIR" remote get-url origin | sed -E 's#.*github\.com[/:]([^/]+)/.*#\1#')
  BRANCH="release/v${VERSION}"

  echo "Looking up open PR from ${FORK_OWNER}:${BRANCH} on ${UPSTREAM_REPO}..."
  PR_REF=$(gh pr list \
    --repo "$UPSTREAM_REPO" \
    --state open \
    --head "${FORK_OWNER}:${BRANCH}" \
    --json number \
    --jq '.[0].number // empty')
  if [[ -z "$PR_REF" ]]; then
    echo "error: no open PR found for ${FORK_OWNER}:${BRANCH} on ${UPSTREAM_REPO}" >&2
    exit 1
  fi
  echo "Found PR #${PR_REF}"
fi

echo "Watching CI checks on ${UPSTREAM_REPO}#${PR_REF} (can take 20-40 min)..."
gh pr checks "$PR_REF" --repo "$UPSTREAM_REPO" --watch --fail-fast

echo "Checks passed; merging..."
gh pr merge "$PR_REF" --repo "$UPSTREAM_REPO" --merge --delete-branch

echo "Merged ${UPSTREAM_REPO}#${PR_REF}"
