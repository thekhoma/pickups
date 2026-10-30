#!/usr/bin/env bash
# Rebuilds the demo-mode media in web/public/mock from a Pickups library (real shots, takes, receipts and cast; see
# make_mock_media.py, which says which library takes it reads). Needs ffmpeg and that library (PICKUPS_HOME).
#
#   npm run mock-media                       (all of it)
#   npm run mock-media -- videos stills      (some of it: videos, stills, people, share)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
exec uv run --project "$HERE/../.." python "$HERE/make_mock_media.py" "$@"
