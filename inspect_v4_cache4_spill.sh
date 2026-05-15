#!/usr/bin/env bash
set -euo pipefail

CACHE_DIR="${CACHE_DIR:-${TRITON_CACHE_DIR:-/tmp/triton-v4-cache}}"
PYTHON_BIN="${PYTHON_BIN:-/home/lishengping/miniconda3/bin/python}"
CUOBJDUMP="${CUOBJDUMP:-/usr/local/cuda-12.2/bin/cuobjdump}"
NVDISASM="${NVDISASM:-/usr/local/cuda-12.2/bin/nvdisasm}"

shopt -s nullglob

for cubin in "${CACHE_DIR}"/*/_dc_onekernel_cache4.cubin; do
  json="${cubin%.cubin}.json"
  if [[ ! -f "${json}" ]]; then
    continue
  fi

  meta="$("${PYTHON_BIN}" -c 'import json, sys
d = json.load(open(sys.argv[1]))
print("warps={} stages={} shared={}".format(d.get("num_warps"), d.get("num_stages"), d.get("shared")))' "${json}")"
  usage="$("${CUOBJDUMP}" --dump-resource-usage "${cubin}" | awk '/Function _dc_onekernel_cache4/{getline; print}')"
  local_sass="$("${NVDISASM}" "${cubin}" | grep -Ec '\b(STL|LDL)')"

  printf '%-48s | %s | %s | local_sass=%s\n' "$(basename "$(dirname "${cubin}")")" "${meta}" "${usage}" "${local_sass}"
done
