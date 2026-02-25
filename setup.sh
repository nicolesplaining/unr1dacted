#!/usr/bin/env bash
set -euo pipefail

# Installs Miniconda (if needed) and creates/updates the conda env from environment.yml.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Support running this script from either repo root (`setup.sh`) or `scripts/`.
if [[ -f "${SCRIPT_DIR}/environment.yml" ]]; then
  PROJECT_ROOT="${SCRIPT_DIR}"
elif [[ -f "${SCRIPT_DIR}/../environment.yml" ]]; then
  PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
else
  PROJECT_ROOT="${SCRIPT_DIR}"
fi
ENV_FILE="${PROJECT_ROOT}/environment.yml"
MINICONDA_DIR="${MINICONDA_DIR:-$HOME/miniconda3}"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Error: environment.yml not found at ${ENV_FILE}"
  exit 1
fi

ARCH="$(uname -m)"
case "${ARCH}" in
  x86_64) MINICONDA_ARCH="x86_64" ;;
  aarch64|arm64) MINICONDA_ARCH="aarch64" ;;
  *)
    echo "Unsupported architecture: ${ARCH}"
    exit 1
    ;;
esac

INSTALLER="Miniconda3-latest-Linux-${MINICONDA_ARCH}.sh"
INSTALLER_PATH="${PROJECT_ROOT}/${INSTALLER}"
INSTALLER_URL="https://repo.anaconda.com/miniconda/${INSTALLER}"

if [[ ! -x "${MINICONDA_DIR}/bin/conda" ]]; then
  echo "Miniconda not found at ${MINICONDA_DIR}. Installing..."
  if command -v wget >/dev/null 2>&1; then
    wget -O "${INSTALLER_PATH}" "${INSTALLER_URL}"
  elif command -v curl >/dev/null 2>&1; then
    curl -fsSL -o "${INSTALLER_PATH}" "${INSTALLER_URL}"
  else
    echo "Error: need wget or curl to download Miniconda installer."
    exit 1
  fi

  bash "${INSTALLER_PATH}" -b -p "${MINICONDA_DIR}"
  rm -f "${INSTALLER_PATH}"
fi

echo "Initializing conda for bash..."
"${MINICONDA_DIR}/bin/conda" init bash >/dev/null 2>&1 || true

# Load conda in this shell without needing a new terminal.
source "${MINICONDA_DIR}/etc/profile.d/conda.sh"

ENV_NAME="$(awk '/^name:[[:space:]]*/ {print $2; exit}' "${ENV_FILE}")"
if [[ -z "${ENV_NAME}" ]]; then
  echo "Error: could not read env name from ${ENV_FILE}."
  exit 1
fi

if conda env list | awk '{print $1}' | grep -Fx "${ENV_NAME}" >/dev/null 2>&1; then
  echo "Updating existing env '${ENV_NAME}' from ${ENV_FILE}..."
  conda env update -f "${ENV_FILE}" --prune
else
  echo "Creating env '${ENV_NAME}' from ${ENV_FILE}..."
  conda env create -f "${ENV_FILE}"
fi

echo
echo "Done."
echo "Activate with: conda activate ${ENV_NAME}"
