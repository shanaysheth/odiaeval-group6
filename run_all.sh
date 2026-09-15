#!/usr/bin/env bash
# Full pipeline, start to finish. ~45 min on a single T4.
set -euo pipefail
cd "$(dirname "$0")/src"
python build_dataset.py     # stages 1-7: acquire -> splits
python finetune.py          # stage 8: fine-tune + test
python analysis.py          # stage 9: diagnostics + native harness
