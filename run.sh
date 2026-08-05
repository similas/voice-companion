#!/usr/bin/env bash
# Launch the companion. Keeps you from having to remember the venv path.
set -euo pipefail
cd "$(dirname "$0")"

VENV="$HOME/.venvs/voice-companion"
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "venv missing at $VENV — see README.md (Install)" >&2
  exit 1
fi

# The venv deliberately lives OUTSIDE this directory. nltk 3.10 refuses to import
# any module located under the current working directory, so a venv at ./.venv
# would make every dependency unimportable. See README "Why the venv is elsewhere".
exec "$VENV/bin/python" -m app.main "$@"
