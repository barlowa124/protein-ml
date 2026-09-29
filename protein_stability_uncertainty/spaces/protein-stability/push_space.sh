#!/usr/bin/env bash
# Copy the Space files into a cloned HF Space repo and push.
# Usage: ./push_space.sh /path/to/cloned/space-repo
set -euo pipefail
DST="${1:?path to cloned huggingface space repo}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cp "$SCRIPT_DIR/app.py" "$SCRIPT_DIR/requirements.txt" "$SCRIPT_DIR/README.md" "$DST/"
cp "$REPO_ROOT/results/deploy_esm2.json" "$DST/deploy_esm2.json"

cd "$DST"
git add -A
git status --short
echo "Review the staged files, then: git commit -m 'protein Tm conformal space' && git push"
