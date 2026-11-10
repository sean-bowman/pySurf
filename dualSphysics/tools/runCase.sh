#!/usr/bin/env bash
# -- Run one DualSPHysics case: GenCase, then the GPU solver -- #
#
#     bash runCase.sh <case directory> <case name> [gencase]
#
# The directory holds <name>_Def.xml (dsphCase.py writes it). With 'gencase' only GenCase runs, to
# check the particles. Output lands in <case directory>/out; logs beside it.
#
# Sean Bowman [11/08/2026]
set -euo pipefail
BIN=$HOME/externalReferences/dualsphysics/bin/linux
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:$BIN
cd "$1"
NAME=$2
rm -rf out
"$BIN/GenCase_linux64" ${NAME}_Def out/$NAME -save:all > gencase.log 2>&1 || { tail -20 gencase.log; exit 1; }
grep -E "Particles|Fluid|Bound|Moving" gencase.log | head -8
if [ "${3:-}" = gencase ]; then exit 0; fi
start=$(date +%s)
"$BIN/DualSPHysics5.4_linux64" -gpu out/$NAME out > run.log 2>&1 || { tail -20 run.log; exit 1; }
echo "solver $(( $(date +%s) - start )) s"
echo "RUN-END" >> run.log
