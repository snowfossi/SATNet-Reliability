import numpy as np
import torch
import pickle
import math
import mixnet

for k in [2, 4, 8, 16]:
    dict = {'input': [], 'is_input': [], 'output': [], 'label': []}
    dict['description'] = 'input: previous modulo(log k), digit(0/1), current modulo(log k), output: 1, 2log(k)+1, aux, label: input, counting modulo k'
    nbit = math.ceil(math.log(k, 2))
    model = mixnet.MixNet(1 + 2 * nbit, aux=k//2)
    S = torch.load(f'logs/{k}_modulo-satnet-sparsity-False-aux{k//2}-m10-lr0.04-bsz10/it20.pth')['S']
    model.C.data = S @ S.T
    model.C.data.fill_diagonal_(0)
    model = model.cuda()
    for i in range(k):
        is_input = torch.cat((torch.ones((1, 1 + nbit), dtype=torch.int32), torch.zeros((1, nbit), dtype=torch.int32)), dim=1).to('cuda')
        prev_modulo = torch.zeros((1, nbit), device='cuda')
        binary = bin(i)[2:].zfill(nbit)
        binary = [float(digit) for digit in binary]
        prev_modulo[0] = torch.tensor(binary).to('cuda')

        data = torch.cat((prev_modulo, torch.ones(1, 1).to('cuda'), torch.zeros(1, nbit).to('cuda')), dim=1)
        output = model(data, is_input)
        label = (i + 1) % k
        label = bin(label)[2:].zfill(nbit)
        label = [float(digit) for digit in label]
        label = np.array(label).reshape(1, -1)
        label = np.concatenate((data[:, :1 + nbit].cpu().detach().numpy().reshape(1, -1), label), axis=1)
        print(label)
        dict['output'].append(output.cpu().detach().numpy())
        dict['input'].append(data.cpu().detach().numpy())
        dict['is_input'].append(is_input.cpu().detach().numpy())
        dict['label'].append(label)

        data = torch.cat((prev_modulo, torch.zeros(1, 1).to('cuda'), torch.zeros(1, nbit).to('cuda')), dim=1)
        output = model(data, is_input)
        label = i % k
        label = bin(label)[2:].zfill(nbit)
        label = [float(digit) for digit in label]
        label = np.array(label).reshape(1, -1)
        label = np.concatenate((data[:, :1 + nbit].cpu().detach().numpy().reshape(1, -1), label), axis=1)

        dict['output'].append(output.cpu().detach().numpy())
        dict['input'].append(data.cpu().detach().numpy())
        dict['is_input'].append(is_input.cpu().detach().numpy())
        dict['label'].append(label)
    with open(f'verifications/stream/data/{k}_modulo_all_pos.pickle', 'wb') as f:
        pickle.dump(dict, f)
    np.save(f'verifications/stream/weights/{k}_modulo_c_n_{1 + 2 * nbit}_aux_{k//2}.npy', model.C.data.cpu().detach().numpy())