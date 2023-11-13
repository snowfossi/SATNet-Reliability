#!/bin/bash
python3 mix2sat.py --cmatrix C_sparse.npy --aux 0 --fout g_sat_common.txt

python3 parallel.py --start 0 --length 16 > a_out.txt
for i in {1..5351}; do 
    s=$(($i*16))
    python3 parallel.py --start "$s" --length 16 >> a_out.txt
done