#!/bin/bash
cd "$(dirname "$0")"
for s in 0 1 2; do for st in none ros smote borderline; do
 python3 -m hai_dwf.components --dataset edge_rare --seed $s --cfg config/t21_$st.yaml >> t21.log 2>&1
 python3 -m hai_dwf.fusion --dataset edge_rare --seed $s --cfg config/t21_$st.yaml >> t21.log 2>&1
 echo "done $st $s" >> t21.progress
done; done; echo ALLDONE >> t21.progress
