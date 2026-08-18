#!/bin/bash
# run_twop_alone.sh — run TwoPunctureABE standalone (timed) in a scratch
# copy of the output directory, without touching GW250118.
set -u
cd /home/h3240105662/hpc101-nogit/src/lab4

SCRATCH=/tmp/twop_run
rm -rf "$SCRATCH"
cp -r GW250118/AMSS_NCKU_output "$SCRATCH"
cp build_final2/TwoPunctureABE "$SCRATCH/"

cd "$SCRATCH"
echo "== TwoPunctureABE start $(date +%H:%M:%S) cwd=$SCRATCH"
t0=$(date +%s.%N)
./TwoPunctureABE < /dev/null > TwoPunctureABE_out.log 2>&1
rc=$?
t1=$(date +%s.%N)
echo "rc=$rc elapsed=$(echo "$t1 $t0" | awk '{printf "%.2f", $1-$2}')s"
echo "== outputs:"
ls -la Ansorg.psid puncture_parameters_new.txt 2>/dev/null
echo "== log tail:"
tail -5 TwoPunctureABE_out.log
echo "========== DONE =========="
