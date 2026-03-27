#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NATIVE_INC="${ROOT_DIR}/../python_model/native/include"
OUT_LIB="${ROOT_DIR}/libecm_ekf_new.so"

gcc -O2 -fPIC -shared -include math.h \
  -I"${NATIVE_INC}" \
  -I"${ROOT_DIR}" \
  "${ROOT_DIR}/battery_ekf.c" \
  "${ROOT_DIR}/interpolation.c" \
  "${ROOT_DIR}/ECM_parameter.c" \
  -lm \
  -o "${OUT_LIB}"

echo "Built ${OUT_LIB}"
