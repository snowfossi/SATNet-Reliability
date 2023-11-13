#!/bin/bash

python3 mix2sat.py --cmatrix C_sparse.npy --delta delta_c_matrix_one_hot_rules.pt --fout cplusdelta_com.txt

python3 solve_all_gt.py

cd sol

grep "Objective" -r . | awk '{print $5}' | sort | uniq -c


cd ..

python3 max2pbo.py --fin cplusdelta_com.txt --fout cplusdelta_com.opb

cat cplusdelta_com.opb > ranking_w_delta.opb

cat constraints.opb >> ranking_w_delta.opb

gurobi_cl Resultfile=ranking_w_delta.sol ranking_w_delta.opb > ranking_w_delta.log

