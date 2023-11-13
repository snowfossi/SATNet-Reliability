#!/bin/bash

python3 mix2sat.py --cmatrix cmatrix_sudoku_n_729_aux_300.pth --fout mix_common9.txt
python3 check_sub.py > gurobilogs.log