import numpy as np
import torch
import pickle
import os
import math

def convert2bin(x, x_bin, nbits):
    b, l = x.shape
    for i in range(b):
        for j in range(l):
            digits = bin(x[i, j])[2:].zfill(nbits)
            digits = [float(digit) for digit in digits]
            x_bin[i, j] = torch.tensor(digits)

for k in [2, 4, 8, 16]:
    N = 1000
    l = k * 4
    nbits = math.ceil(math.log(k, 2))
    digit = torch.randint(0, 2, size=(N, l))
    modulo = torch.zeros((N, l), dtype=torch.int)
    modulo_binary = torch.zeros((N, l, nbits))
    for i in range(l):
        if i == 0:
            modulo[:, i] = digit[:, i]
        else:
            modulo[:, i] = (modulo[:, i - 1] + digit[:, i]) % k
    convert2bin(modulo, modulo_binary, nbits)
    dict = {'digit': digit, 'modulo': modulo_binary}
    os.makedirs(f'data/k_modulo/{k}', exist_ok=True)
    with open(f'data/k_modulo/{k}/data.pkl', 'wb') as f:
        pickle.dump(dict, f)