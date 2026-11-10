#!/usr/bin/env bash
# -- DualSPHysics Build Check: run the shipped 2D dam break on both solvers -- #
#
# Runs CaseDambreakVal2D as shipped (examples/main/01_DamBreak) through GenCase and then the GPU and
# the CPU solver (OpenMP capped at 6 threads), and writes each run's fluid particles as PartVTK CSV
# frames. checkDamBreak.py compares the two runs and the experiment.
#
#     bash runDamBreak.sh
#
# Runs land in ~/dsphRuns/dambreakVal2D_gpu and _cpu. On the reference machine the GPU run took
# 144 s and the CPU run 500 s.
#
# Sean Bowman [11/08/2026]
set -euo pipefail
DS=$HOME/externalReferences/dualsphysics
BIN=$DS/bin/linux
RUNS=$HOME/dsphRuns
CASE=CaseDambreakVal2D
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:$BIN
for mode in gpu cpu; do
    dir=$RUNS/dambreakVal2D_$mode
    rm -rf "$dir"
    mkdir -p "$dir"
    cp "$DS/examples/main/01_DamBreak/${CASE}_Def.xml" "$dir/"
    cd "$dir"
    "$BIN/GenCase_linux64" ${CASE}_Def out/$CASE -save:all > gencase.log 2>&1
    start=$(date +%s)
    if [ "$mode" = gpu ]; then
        "$BIN/DualSPHysics5.4_linux64" -gpu out/$CASE out > run.log 2>&1
    else
        "$BIN/DualSPHysics5.4CPU_linux64" -cpu -ompthreads:6 out/$CASE out > run.log 2>&1
    fi
    echo "$mode solver $(( $(date +%s) - start )) s"
    "$BIN/PartVTK_linux64" -dirdata out/data -savecsv csv/PartFluid -onlytype:-all,fluid -vars:-all > partvtk.log 2>&1
    echo "$mode csv files: $(ls csv | wc -l)"
done
