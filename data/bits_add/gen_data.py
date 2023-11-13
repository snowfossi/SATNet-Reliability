import torch
import pickle
import numpy as np


N = 4000
a = torch.randint(0, 2, (N, 9)).float()
b = torch.randint(0, 2, (N, 9)).float()
a[:, 8] = 0
b[:, 8] = 0
c = torch.zeros((N, 9))
carry = torch.zeros((N, 9))
for i in range(9):
    if i != 0:
        carry[:, i] = (a[:, i] + b[:, i] + carry[:, i - 1]) // 2
        c[:, i] = (a[:, i] + b[:, i] + carry[:, i - 1]) % 2
    else:
        carry[:, i] = (a[:, i] + b[:, i]) // 2
        c[:, i] = (a[:, i] + b[:, i]) % 2

dict = {'a': a, 'b': b, 'c': c, 'carry': carry}

with open('data/bits_add/data.pkl', 'wb') as f:
    pickle.dump(dict, f)