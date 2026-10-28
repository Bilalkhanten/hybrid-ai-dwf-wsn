#!/bin/bash
cd "$(dirname "$0")"
F="grep -v oneDNN|I0000|WARNING"
r(){ python3 -m hai_dwf.components --dataset $2 --seed $3 --cfg config/$1.yaml >> ext.log 2>&1; python3 -m hai_dwf.fusion --dataset $2 --seed $3 --cfg config/$1.yaml >> ext.log 2>&1; echo "done $1 $2 $3" >> ext.progress; }
for s in 0 1 2; do
 for st in none ros smote borderline; do r t21_$st edge_rare $s; done
 r t23_raw nf_ton_iot_v2 $s; r t23_raw wsn_ds $s; r t23_nowsn wsn_ds $s
done
echo ALLDONE >> ext.progress
