import argparse
import time
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm
import satnet
import mixnet


def print_header(msg):
    print('===>', msg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/k_modulo')
    parser.add_argument('--batchSz', type=int, default=10)
    parser.add_argument('--testBatchSz', type=int, default=10)
    parser.add_argument('--aux', type=int, default=1)
    parser.add_argument('--m', type=int, default=10)
    parser.add_argument('--nEpoch', type=int, default=40)
    parser.add_argument('--testPct', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=4e-2)
    parser.add_argument('--save', type=str)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--model', type=str, default='satnet')
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--sparsity', action='store_true', default=False)
    parser.add_argument('--k', type=int)

    args = parser.parse_args()

    # For debugging: fix the random seed
    # npr.seed(1)
    # torch.manual_seed(7)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        print('Using', torch.cuda.get_device_name(0))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.init()

    save = '{}_modulo-{}-sparsity-{}-aux{}-m{}-lr{}-bsz{}'.format(args.k, args.model, args.sparsity, args.aux, args.m, args.lr, args.batchSz)
    if args.save: save = '{}-{}'.format(args.save, save)
    save = os.path.join('logs', save)

    os.makedirs(save, exist_ok=True)

    print_header('Loading data')
    data_dir = os.path.join(args.data_dir, str(args.k))
    with open(os.path.join(data_dir, 'data.pkl'), 'rb') as f:
        dict = pickle.load(f)
    digit, modulo = dict['digit'], dict['modulo']
    N = digit.size(0)
    nbit = modulo.shape[2]
    print("number of all data:", N)
    nTrain = int(N * (1. - args.testPct))

    print_header('Forming inputs')
    is_input = torch.cat((torch.ones((N, 1 + nbit), dtype=torch.int32), torch.zeros((N, nbit), dtype=torch.int32)), dim=1)

    if args.cuda: digit, modulo, is_input = digit.cuda(), modulo.cuda(), is_input.cuda()
    train_set = TensorDataset(digit[:nTrain], modulo[:nTrain], is_input[:nTrain])
    test_set = TensorDataset(digit[nTrain:], modulo[nTrain:], is_input[nTrain:])

    print_header('Building model')
    if args.model == 'mixnet':
        model = mixnet.MixNet(1 + 2 * nbit, aux=args.aux)
    else:
        model = satnet.SATNet(1 + 2 * nbit, m=args.m, aux=args.aux)

    if args.cuda: model = model.cuda()

    if args.pretrained is not None:
        print(model.load_state_dict(torch.load(args.pretrained)))

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    log_file = os.path.join(save, f'{cur_time}.txt')

    if args.model == 'mixnet' and args.sparsity:
        old_C = model.C.data.clone()
    for epoch in range(1, args.nEpoch+1):
        train(epoch, model, optimizer, train_set, args.batchSz, log_file)
        if args.model == 'mixnet' and args.sparsity and torch.norm(model.C.data - old_C) < 0.01 and torch.count_nonzero(model.C.data) > model.data.numel() / 2:
            nonzero_idx = torch.nonzero(model.C.data, as_tuple=True)
            min_v = torch.min(torch.abs(model.C.data[nonzero_idx]))
            mask = (torch.abs(model.C.data) != min_v)
            model.C.data = model.model.C.data * mask
            old_C = model.C.data.clone()
        test(epoch, model, optimizer, test_set, args.testBatchSz, log_file)
        torch.save(model.state_dict(), os.path.join(save, 'it'+str(epoch)+'.pth'))


def run(epoch, model, optimizer, dataset, batchSz, to_train=False, log_file=None):
    loss_final, err_final = 0, 0
    loader = DataLoader(dataset, batch_size=batchSz)
    tloader = tqdm(enumerate(loader), total=len(loader))

    for i, (digit, modulo, is_input) in tloader:
        N, l, nbit = modulo.size()
        # parallel training
        if to_train:
            digit = digit.reshape(N * l, 1)
            previous_modulo = torch.cat((torch.zeros((N, 1, nbit), device=digit.device), modulo[:, :-1]), dim=1)
            previous_modulo = previous_modulo.reshape(N * l ,nbit)
            data = torch.cat((digit, previous_modulo, torch.zeros((N * l, nbit), device=digit.device)), dim=1) # N * l, 1 + 2nbit
            is_input = is_input.unsqueeze(1).repeat(1, l, 1).reshape(N * l, 1 + 2 * nbit)
            output = model(data, is_input)
            pred_modulo = output[:, -nbit:]
            loss = nn.functional.binary_cross_entropy(pred_modulo, modulo.reshape(N * l, nbit))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # sequential testing
        else:
            pred_modulo = torch.zeros((N, l, nbit), device=digit.device)
            pred_modulo_proj = torch.zeros((N, l, nbit), device=digit.device)
            for i in range(l):
                if i == 0:
                    data = torch.cat((digit[:, [i]], torch.zeros((N, 2 * nbit), device=digit.device)), dim=1)
                else:
                    data = torch.cat((digit[:, [i]], pred_modulo_proj[:, i - 1], torch.zeros((N, nbit), device=digit.device)), dim=1)
                output = model(data, is_input)
                pred_modulo[:, i] = output[:, -nbit:]
                pred_modulo_proj[:, i] = torch.where(pred_modulo[:, i] >= 0.5, 1., 0.).detach()
            loss = nn.functional.binary_cross_entropy(pred_modulo, modulo)

        preds = torch.where(pred_modulo >= 0.5, 1., 0.).reshape(N * l, nbit).detach()
        labels = modulo.reshape(N * l, nbit)
        err = 1 - torch.sum(torch.all(preds == labels, dim=1)) / preds.size(0)
        tloader.set_description(
            'Epoch {} {} Loss {:.4f} Err: {:.4f}'.format(epoch, ('Train' if to_train else 'Test '), loss.item(), err))
        loss_final += loss.item()
        err_final += err

    loss_final, err_final = loss_final/len(loader), err_final/len(loader)
    with open(log_file, 'a') as f:
        pre = 'Train' if to_train else 'Test'
        print(f'{pre} epoch:{epoch}, loss={loss_final}, err={err_final}', file=f)
    if not to_train:
        print('TESTING SET RESULTS: Average loss: {:.4f} Err: {:.4f}'.format(loss_final, err_final))
    else:
        print('TRAINING SET RESULTS: Average loss: {:.4f} Err: {:.4f}'.format(loss_final, err_final))

    torch.cuda.empty_cache()


def train(epoch, model, optimizer, dataset, batchSz, log_file):
    run(epoch, model, optimizer, dataset, batchSz, True, log_file)


@torch.no_grad()
def test(epoch, model, optimizer, dataset, batchSz, log_file):
    run(epoch, model, optimizer, dataset, batchSz, False, log_file)


if __name__ == '__main__':
    main()