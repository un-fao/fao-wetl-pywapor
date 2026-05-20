#!/usr/bin/env bash
# Bump the version in pyproject.toml, commit, tag, and push.
# Usage: ./tag_release.sh [major|minor|patch]   (default: patch)

# on dev, finish your work, merge to main via PR
# git checkout main && git pull
# ./tag_release.sh patch   # or minor / major

set -euo pipefail

BUMP="${1:-patch}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYPROJECT="pyproject.toml"
if [[ ! -f "$PYPROJECT" ]]; then
  echo "error: $PYPROJECT not found in $SCRIPT_DIR" >&2
  exit 1
fi

# Refuse to release with a dirty tree.
if ! git diff-index --quiet HEAD --; then
  echo "error: working tree has uncommitted changes; commit or stash first" >&2
  git status --short >&2
  exit 1
fi

BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [[ "$BRANCH" != "main" ]]; then
  echo "error: releases must be cut from 'main' (current branch: $BRANCH)" >&2
  exit 1
fi

git fetch --quiet origin main
LOCAL=$(git rev-parse main)
REMOTE=$(git rev-parse origin/main)
if [[ "$LOCAL" != "$REMOTE" ]]; then
  echo "error: local main ($LOCAL) is not in sync with origin/main ($REMOTE); pull or push first" >&2
  exit 1
fi

# Read current version from pyproject.toml (first `version = "..."` under [project]).
CURRENT=$(awk '
  /^\[project\]/ { in_project = 1; next }
  /^\[/          { in_project = 0 }
  in_project && /^version[[:space:]]*=/ {
    match($0, /"[^"]+"/)
    print substr($0, RSTART + 1, RLENGTH - 2)
    exit
  }
' "$PYPROJECT")

if [[ -z "$CURRENT" ]]; then
  echo "error: could not read version from $PYPROJECT" >&2
  exit 1
fi

if [[ ! "$CURRENT" =~ ^([0-9]+)\.([0-9]+)\.([0-9]+)$ ]]; then
  echo "error: current version '$CURRENT' is not in MAJOR.MINOR.PATCH form" >&2
  exit 1
fi

MAJOR="${BASH_REMATCH[1]}"
MINOR="${BASH_REMATCH[2]}"
PATCH="${BASH_REMATCH[3]}"

case "$BUMP" in
  major) MAJOR=$((MAJOR + 1)); MINOR=0; PATCH=0 ;;
  minor) MINOR=$((MINOR + 1)); PATCH=0 ;;
  patch) PATCH=$((PATCH + 1)) ;;
  *) echo "Usage: $0 [major|minor|patch]" >&2; exit 1 ;;
esac

NEW_VERSION="${MAJOR}.${MINOR}.${PATCH}"
NEW_TAG="v${NEW_VERSION}"

if git rev-parse "$NEW_TAG" >/dev/null 2>&1; then
  echo "error: tag $NEW_TAG already exists" >&2
  exit 1
fi

echo "Bumping $CURRENT -> $NEW_VERSION (tag $NEW_TAG)"

# In-place edit of the first `version = "..."` line inside [project].
python3 - "$PYPROJECT" "$NEW_VERSION" <<'PY'
import re, sys
path, new_version = sys.argv[1], sys.argv[2]
src = open(path).read()
pattern = re.compile(r'(\[project\][^\[]*?\nversion\s*=\s*")[^"]+(")', re.DOTALL)
new, n = pattern.subn(rf'\g<1>{new_version}\g<2>', src, count=1)
if n != 1:
    sys.exit("failed to rewrite version in pyproject.toml")
open(path, "w").write(new)
PY

git add "$PYPROJECT"
git commit -m "Release ${NEW_TAG}"
git tag -a "$NEW_TAG" -m "Release ${NEW_TAG}"
git push origin "$BRANCH"
git push origin "$NEW_TAG"

echo "Released $NEW_TAG"
