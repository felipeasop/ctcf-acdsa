#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PIPELINE_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
export PYTHONDONTWRITEBYTECODE=1
source "$SCRIPT_DIR/lib/common.sh"
MODE=full
case ${1:-} in
    "") ;;
    --analysis-only) MODE=analysis; shift ;;
    *) die "Usage: bash 00_preflight.sh [--analysis-only]" ;;
esac
[[ $# -eq 0 ]] || die "Usage: bash 00_preflight.sh [--analysis-only]"
require_commands python3
if [[ $MODE == full ]]; then
    require_commands wget md5sum gzip samtools bedtools meme fasta-get-markov tomtom
fi
for script in "$SCRIPT_DIR"/*.sh "$SCRIPT_DIR"/lib/*.sh; do
    bash -n "$script"
done
python3 - "$SCRIPT_DIR" <<'PY'
import ast
import pathlib
import sys
import numpy
import sklearn
print(f'Python {sys.version.split()[0]}; NumPy {numpy.__version__}')
print(f'scikit-learn {sklearn.__version__}')
for path in pathlib.Path(sys.argv[1]).rglob('*.py'):
    ast.parse(path.read_text(), filename=str(path))
print('Python and Bash syntax: OK')
PY
python3 -m unittest discover -s "$PIPELINE_DIR/tests" -p 'test_*.py'
if command -v shellcheck >/dev/null 2>&1; then
    shellcheck "$SCRIPT_DIR"/*.sh "$SCRIPT_DIR"/lib/*.sh
else
    printf 'ShellCheck: not installed; shell lint skipped.\n'
fi
