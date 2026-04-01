#!/usr/bin/env bash

set -euo pipefail

repo_root() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  cd "$script_dir/../.." && pwd
}

enter_repo_root() {
  cd "$(repo_root)"
}

require_env() {
  local var_name="$1"
  : "${!var_name:?Set ${var_name} before running this script.}"
}
