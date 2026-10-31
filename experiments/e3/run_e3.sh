#!/usr/bin/env bash
# 一键运行 E3：pilot 默认3个重复，formal 默认10个重复。
set -euo pipefail

script_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
stage="${1:-pilot}"
if [[ $# -gt 0 ]]; then
    shift
fi

case "$stage" in
    pilot|formal) ;;
    *)
        echo "用法: bash experiments/e3/run_e3.sh {pilot|formal} [batch 参数]" >&2
        exit 2
        ;;
esac

exec "$script_root/.venv/bin/python" -B -m experiments.e3.runners.run_e3_batch \
    --stage "$stage" --execute "$@"
