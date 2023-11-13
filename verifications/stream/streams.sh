#!/bin/bash

python3 verify_stream.py --cmatrix 'weights/addition_c_n_5_aux_2.npy' --data 'data/addition_full_space.pickle' --aux 2 --name 'add' --n 5
python3 verify_stream.py --cmatrix 'weights/2_modulo_c_n_3_aux_1.npy' --data 'data/2_modulo_all_pos.pickle' --aux 1 --name 'mod2' --n 3
python3 verify_stream.py --cmatrix 'weights/4_modulo_c_n_5_aux_2.npy' --data 'data/4_modulo_all_pos.pickle' --aux 2 --name 'mod4' --n 5
python3 verify_stream.py --cmatrix 'weights/8_modulo_c_n_7_aux_4.npy' --data 'data/8_modulo_all_pos.pickle' --aux 4 --name 'mod8' --n 7
python3 verify_stream.py --cmatrix 'weights/16_modulo_c_n_9_aux_8.npy' --data 'data/16_modulo_all_pos.pickle' --aux 8 --name 'mod16' --n 9
