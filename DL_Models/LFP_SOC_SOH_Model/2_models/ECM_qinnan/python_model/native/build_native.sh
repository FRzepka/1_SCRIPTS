#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NATIVE_DIR="${ROOT_DIR}/python_model/native"
INC_DIR_NATIVE="${NATIVE_DIR}/include"
INC_DIR_ECHO="${ROOT_DIR}/echo/CM7/Core/Inc"
SRC_DIR_ECHO="${ROOT_DIR}/echo/CM7/Core/Src"
OUT_LIB="${NATIVE_DIR}/libecm_ekf.so"

gcc -O2 -fPIC -shared -include math.h \
  -I"${INC_DIR_NATIVE}" \
  -I"${INC_DIR_ECHO}" \
  "${SRC_DIR_ECHO}/battery_ekf.c" \
  "${SRC_DIR_ECHO}/interpolation.c" \
  "${SRC_DIR_ECHO}/ECM_parameter.c" \
  -lm \
  -o "${OUT_LIB}"

echo "Built ${OUT_LIB}"
