#!/bin/bash

abs_file="$1"
pkg_root="src"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv_python="$repo_root/.venv/bin/python"

if [[ ! -x "$venv_python" ]]; then
  echo "❌ .venv/bin/python not found or not executable"
  exit 1
fi

rel_path="${abs_file#$repo_root/}"
rel_path="${rel_path%.py}"

if [[ "$rel_path" != $pkg_root/* ]]; then
  echo "❌ Error: file is not under package root '$pkg_root'"
  exit 1
fi

mod_path="${rel_path//\//.}"

# ✅ DEBUG INFO
echo "▶ Using Python: $venv_python"
"$venv_python" --version
"$venv_python" -c "import sys; print('VENV site-packages:', sys.prefix)"

# ✅ Run your module
echo "▶ Running module: $mod_path"
"$venv_python" -m "$mod_path"