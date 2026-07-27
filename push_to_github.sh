#!/usr/bin/env bash
set -euo pipefail

REMOTE_URL="https://github.com/mpetalcorin/retail-decision-intelligence.git"

git remote remove origin 2>/dev/null || true
git remote add origin "$REMOTE_URL"
git branch -M main
git push -u origin main
