#!/usr/bin/env bash
# Link this repo's skills into ~/.claude/skills so they work from any directory.
#
# Symlinks rather than copies: research happens in whatever folder the vault
# lives in, but the skills are edited here, and a copy would mean every fix
# needs reinstalling before it takes effect.

set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/skills"
DEST="${CLAUDE_SKILLS_DIR:-$HOME/.claude/skills}"

usage() {
    cat <<'EOF'
usage: ./install.sh [--force] [--uninstall] [--dry-run]

  --force      replace an existing directory or link of the same name
  --uninstall  remove only the links that point into this repo
  --dry-run    print what would happen, change nothing

Set CLAUDE_SKILLS_DIR to install somewhere other than ~/.claude/skills.
EOF
}

force=0
uninstall=0
dry=0
for arg in "$@"; do
    case "$arg" in
        --force) force=1 ;;
        --uninstall) uninstall=1 ;;
        --dry-run|-n) dry=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $arg" >&2; usage >&2; exit 2 ;;
    esac
done

run() { if [ "$dry" -eq 1 ]; then echo "  would: $*"; else "$@"; fi; }

if [ ! -d "$SRC" ]; then
    echo "no skills/ directory next to install.sh (looked in $SRC)" >&2
    exit 1
fi

mkdir -p "$DEST"
echo "skills:  $SRC"
echo "target:  $DEST"
echo

linked=0 skipped=0 removed=0

for path in "$SRC"/*/; do
    name="$(basename "$path")"
    target="$DEST/$name"

    if [ "$uninstall" -eq 1 ]; then
        # Only ever remove a symlink that points back into this repo. Deleting
        # a real directory here would destroy a skill someone wrote by hand.
        if [ -L "$target" ] && [ "$(readlink "$target")" = "${path%/}" ]; then
            run rm "$target"
            echo "  removed  $name"
            removed=$((removed + 1))
        fi
        continue
    fi

    if [ -L "$target" ] && [ "$(readlink "$target")" = "${path%/}" ]; then
        echo "  ok       $name (already linked)"
        linked=$((linked + 1))
        continue
    fi

    if [ -e "$target" ] || [ -L "$target" ]; then
        if [ "$force" -eq 1 ]; then
            run rm -rf "$target"
        else
            echo "  SKIP     $name — something else is already there. --force to replace." >&2
            skipped=$((skipped + 1))
            continue
        fi
    fi

    run ln -s "${path%/}" "$target"
    echo "  linked   $name"
    linked=$((linked + 1))
done

echo
if [ "$uninstall" -eq 1 ]; then
    echo "removed $removed link(s)."
    exit 0
fi

echo "$linked linked, $skipped skipped."
if [ "$skipped" -gt 0 ]; then
    echo
    echo "The skipped names already exist in $DEST. If those are the old kb-*"
    echo "style skills these replace, delete them; otherwise rename one side."
fi
cat <<'EOF'

Start a new Claude Code session (skills are discovered at startup), then:

    /research-idea does curriculum ordering help small-model math reasoning?

Or invoke any single step on its own — /survey-literature, /build-vault,
/design-experiment, /lint-vault, and so on.
EOF
