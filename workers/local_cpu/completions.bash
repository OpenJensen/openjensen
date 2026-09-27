# Optional: set OPENJENSEN_REPO to your checkout, then source this file.
# No shell/profile file is modified by the installer.
openjensen-cpu-setup() {
  if [[ -z "${OPENJENSEN_REPO:-}" ]]; then
    printf '%s\n' 'Set OPENJENSEN_REPO to the complete repository path first.' >&2
    return 2
  fi
  python3 "$OPENJENSEN_REPO/workers/local_cpu/manage.py" "$@"
}
_openjensen_cpu_setup_complete() {
  local current="${COMP_WORDS[COMP_CWORD]}"
  COMPREPLY=()
  if [[ "$current" == -* || "$COMP_CWORD" == 1 ]]; then
    while IFS= read -r candidate; do COMPREPLY+=("$candidate"); done < <(
      compgen -W 'plan install verify config --root --python --uv --execute --output --base --help --version' -- "$current"
    )
  fi
}
complete -o default -F _openjensen_cpu_setup_complete openjensen-cpu-setup
