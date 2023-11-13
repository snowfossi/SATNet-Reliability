#!/bin/bash

python3 mix2sat.py --cmatrix sdk4_gt_c.npy --aux 0 --fout common4.txt

echo "globally optimal weight:"

../cashwmaxsatcoreplus -m common4.txt

echo "weights of the 288 valid boards:"

python3 check_space4.py