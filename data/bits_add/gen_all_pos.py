import torch
import numpy as np
import pickle
import mixnet

model = mixnet.MixNet(5, aux=2)
model = model.cuda()
S = torch.load('logs/bit_add-satnet-sparsity-False-aux2-m10-lr0.01-bsz40/it5.pth')['S']
model.C.data = S @ S.T
model.C.data.fill_diagonal_(0)

device = 'cuda'
dict = {'input': [], 'is_input': [], 'output': [], 'label': []}
dict['input'] = [np.array(((0, 0, 0, 0, 0))), np.array(((0, 0, 1, 0, 0))), np.array(((0, 1, 0, 0, 0))), np.array(((0, 1, 1, 0, 0))),
                 np.array(((1, 0, 0, 0, 0))), np.array(((1, 0, 1, 0, 0))), np.array(((1, 1, 0, 0, 0))), np.array(((1, 1, 1, 0, 0)))]
dict['is_input'] = [np.array(((1, 1, 1, 0, 0)))] * 8
dict['label'] = [np.array(((0, 0, 0, 0, 0))), np.array(((0, 0, 1, 1, 0))), np.array(((0, 1, 0, 1, 0))), np.array(((0, 1, 1, 0, 1))),
                 np.array((((1, 0, 0, 1, 0)))), np.array(((1, 0, 1, 0, 1))), np.array(((1, 1, 0, 0, 1))), np.array(((1, 1, 1, 1, 1)))]
dict['description'] = 'input: a b old_carry c new_carry, output: 1 5 aux'
for each in dict['input']:
    each = torch.from_numpy(each).to(device)
    output = model(each.unsqueeze(0), torch.tensor(((1, 1, 1, 0, 0)), device=device).unsqueeze(0)).cpu().detach().numpy()
    dict['output'].append(output)
    print(output)
with open('verifications/stream/data/addition_full_space.pickle', 'wb') as f:
    pickle.dump(dict, f)
np.save('verifications/stream/weights/addition_c_n_5_aux_2.npy', model.C.data.cpu().detach().numpy())