#!/usr/bin/env bash
# Initialize all submodules in a git worktree, borrowing objects from the main checkout via --reference
# so a new worktree costs almost no disk space or network traffic. Run from anywhere inside the worktree.
# Each submodule falls back to a regular clone if the main checkout has not initialized it.
# Note: the worktree then depends on the main checkout's .git/modules; do not delete or deinit it there.
# Warning: submodule checkouts are forced, so re-running this discards uncommitted changes inside submodules.
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
MAIN="$(git -C "$ROOT" worktree list --porcelain | awk 'NR == 1 { print $2 }')"

if [[ "$ROOT" == "$MAIN" ]]; then
  git -C "$ROOT" submodule update --init --recursive
  exit 0
fi

# $1 = path of a (sub)repository relative to $ROOT, empty for the superproject.
# Submodules are handled one level at a time because --recursive would pass the same --reference to nested ones.
# --checkout overrides a custom submodule.<name>.update mode (e.g. merge), which on a fresh clone is a no-op and
# leaves the working tree empty. --force also checks out submodules that are already at the recorded commit, which
# repairs clones left empty or on the wrong commit by an interrupted `git submodule update`.
init_level() {
  local dir="$ROOT${1:+/$1}"
  local subs sub rel
  # Gitlinks from the index rather than .gitmodules, which can list stale entries.
  mapfile -t subs < <(git -C "$dir" ls-files --stage | awk '$1 == "160000"' | cut -f2)
  for sub in "${subs[@]}"; do
    rel="${1:+$1/}$sub"
    if git -C "$MAIN/$rel" rev-parse --git-dir >/dev/null 2>&1 &&
      [[ "$(git -C "$MAIN/$rel" rev-parse --show-toplevel)" == "$MAIN/$rel" ]]; then
      git -C "$dir" submodule update --init --checkout --force --reference "$MAIN/$rel" -- "$sub"
    else
      echo "warning: $rel not initialized in $MAIN, cloning without --reference" >&2
      git -C "$dir" submodule update --init --checkout --force -- "$sub"
    fi
    init_level "$rel"
  done
}

init_level ""
