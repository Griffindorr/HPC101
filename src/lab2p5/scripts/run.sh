#!/usr/bin/env bash

set -e

./scripts/compile.sh

# IME workload must run at CPU 0-3
srun -p riscv -c 8 numactl -C 0-3 ./build/gemm-rv
