#!/usr/bin/env bash
# -- Build DualSPHysics v5.4 (GPU and CPU solvers) in WSL -- #
#
# The DualSPHysics GitHub repository ships GenCase, PartVTK and MeasureTool prebuilt but no solver.
# This builds both solvers from source, in WSL (Ubuntu 24.04, g++ 13) with CUDA 12.8 for the RTX
# 2070 Super (sm_75), from a local clone of the shallow clone kept on Windows, so the build tree has
# Unix line endings and lives on WSL's own disk. The source stays as published:
#   - g++ 13 no longer pulls <cstdint> in through other headers, and JBinaryData.cpp uses SIZE_MAX
#     without it, so the C++ compiler is a wrapper named g++ that forces that one header in
#   - nvcc feeds the host compiler preprocessed intermediates, where a forced <cstdint> collides with
#     the copy already inlined, so nvcc keeps plain g++: the Makefile's release flags with -ccbin set
#   - Chrono (multibody) and MoorDynPlus (moorings) are off; WaveGen, which carries the solitary
#     wave piston, stays on
#
#     bash buildDualSphysics.sh [path to the Windows clone, as WSL sees it]
#
# The clone defaults to externalReferences/dualsphysics in the folder that holds pySurf.
#
# Solvers land in ~/externalReferences/dualsphysics/bin/linux; build logs in ~/externalReferences.
# On the reference machine the GPU solver took 16 minutes, the CPU solver 14.
#
# Sean Bowman [11/08/2026]
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SRC=${1:-$(cd "$HERE/../../.." && pwd)/externalReferences/dualsphysics}
ROOT=$HOME/externalReferences
DST=$ROOT/dualsphysics
CUDA=/usr/local/cuda-12.8
mkdir -p "$ROOT"
if [ ! -d "$DST/.git" ]; then
    git clone -q --depth 1 "file://$SRC" "$DST"
fi
echo "clone: $(git -C "$DST" log -1 --format='%h %cs')"
cd "$DST/src/source"
WRAP=$ROOT/gccWrap
mkdir -p "$WRAP"
printf '#!/usr/bin/env bash\nexec /usr/bin/g++ -include cstdint "$@"\n' > "$WRAP/g++"
chmod +x "$WRAP/g++"
COMMON="COMPILE_CHRONO=NO COMPILE_MOORDYNPLUS=NO CC=$WRAP/g++"
NVFLAGS="-c -gencode=arch=compute_75,code=sm_75 -O3 -ccbin /usr/bin/g++ -D_WITHMR -D_WITHGPU -DCUB_IGNORE_DEPRECATED_CPP_DIALECT -DTHRUST_IGNORE_DEPRECATED_CPP_DIALECT -use_fast_math"
rm -f ./*.o
start=$(date +%s)
make -j6 -f Makefile $COMMON DIRTOOLKIT=$CUDA NCC=$CUDA/bin/nvcc "NCCFLAGS=$NVFLAGS" > "$ROOT/buildGpu.log" 2>&1 || { echo GPU-BUILD-FAILED; grep -m 20 -E 'error' "$ROOT/buildGpu.log"; exit 1; }
echo "gpu build $(( $(date +%s) - start )) s"
start=$(date +%s)
make -j6 -f Makefile_cpu $COMMON > "$ROOT/buildCpu.log" 2>&1 || { echo CPU-BUILD-FAILED; grep -m 20 -E 'error' "$ROOT/buildCpu.log"; exit 1; }
echo "cpu build $(( $(date +%s) - start )) s"
ls -la "$DST/bin/linux"
