#!/usr/bin/env python3
#
# Partly derived from:
#   https://github.com/locuslab/optnet/blob/master/sudoku/train.py

import argparse
import time
import os
import shutil
import csv
import pickle
import numpy as np
import numpy.random as npr
# import setproctitle
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm
# from torch.profiler import profile, record_function, ProfilerActivity

import satnet
import mixnet
import ast


def SudokuSolver(boardSz, model, aux=0, m=0):
    n = boardSz ** 6
    if model == 'satnet':
        return satnet.SATNet(n, m, aux)
    else:
        return mixnet.MixNet(n, aux=aux)


class DigitConv(nn.Module):
    '''
    Convolutional neural network for MNIST digit recognition. From:
    https://github.com/pytorch/examples/blob/master/mnist/main.py
    '''

    def __init__(self):
        super(DigitConv, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)[:, :9].contiguous()


class MNISTSudokuSolver(nn.Module):
    def __init__(self, boardSz, model, aux, m):
        super(MNISTSudokuSolver, self).__init__()
        self.digit_convnet = DigitConv()
        self.sudoku_solver = SudokuSolver(boardSz, model, aux, m)
        self.boardSz = boardSz
        self.nSq = boardSz ** 2

    def forward(self, x, is_inputs):
        nBatch = x.shape[0]
        x = x.flatten(start_dim=0, end_dim=1)
        digit_guess = self.digit_convnet(x)
        puzzles = digit_guess.view(nBatch, self.nSq * self.nSq * self.nSq)

        solution = self.sudoku_solver(puzzles, is_inputs)
        return solution


class CSVLogger(object):
    def __init__(self, fname):
        self.f = open(fname, 'w')
        self.logger = csv.writer(self.f)

    def log(self, fields):
        self.logger.writerow(fields)
        self.f.flush()


class FigLogger(object):
    def __init__(self, fig, base_ax, title):
        self.colors = ['tab:red', 'tab:blue']
        self.labels = ['Loss (entropy)', 'Error']
        self.markers = ['d', '.']
        self.axes = [base_ax, base_ax.twinx()]
        base_ax.set_xlabel('Epochs')
        base_ax.set_title(title)

        for i, ax in enumerate(self.axes):
            ax.set_ylabel(self.labels[i], color=self.colors[i])
            ax.tick_params(axis='y', labelcolor=self.colors[i])

        self.reset()
        self.fig = fig

    def log(self, args):
        for i, arg in enumerate(args[-2:]):
            self.curves[i].append(arg)
            x = list(range(len(self.curves[i])))
            self.axes[i].plot(x, self.curves[i], self.colors[i], marker=self.markers[i])
            self.axes[i].set_ylim(0, 1.05)

        self.fig.canvas.draw()

    def reset(self):
        for ax in self.axes:
            for line in ax.lines:
                line.remove()
        self.curves = [[], []]


def print_header(msg):
    print('===>', msg)


def find_unperm(perm):
    unperm = torch.zeros_like(perm)
    for i in range(perm.size(0)):
        unperm[perm[i]] = i
    return unperm


def decode_mask(m):
    l = []
    while m > 0:
        l.append(1 if m % 2 else 0)
        m = m // 2
    l.extend([0] * (16 - len(l)))
    l.reverse()
    return l


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--boardSz', type=int, default=2)
    parser.add_argument('--batchSz', type=int, default=40)
    parser.add_argument('--testBatchSz', type=int, default=40)
    parser.add_argument('--aux', type=int, default=0)
    parser.add_argument('--m', type=int, default=50)
    parser.add_argument('--nEpoch', type=int, default=100)
    parser.add_argument('--testPct', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--lr_step_size', type=int, default=20, help='Learning rate step size')
    parser.add_argument('--lr_factor', type=float, default=0.1, help='Learning rate factor')
    parser.add_argument('--save', type=str)
    parser.add_argument('--no_cuda', action='store_true') # , default=True
    parser.add_argument('--mnist', action='store_true')
    parser.add_argument('--perm', action='store_true')
    parser.add_argument('--model', type=str, default='satnet')
    parser.add_argument('--sparse', action='store_true', default=False)
    parser.add_argument('--pretrained', type=str, default=None) # 'logs/sudoku4.boardSz2-mixnet-aux0-m50-lr0.002-bsz40/it5.pth'

    args = parser.parse_args()

    # For debugging: fix the random seed
    npr.seed(1)
    torch.manual_seed(7)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    np.random.seed(1)
    random.seed(1)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        print('Using', torch.cuda.get_device_name(0))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.init()

    save = 'sudoku4{}{}.boardSz{}-{}-aux{}-m{}-lr{}-bsz{}'.format(
        '.perm' if args.perm else '', '.mnist' if args.mnist else '',
        args.boardSz, args.model, args.aux, args.m, args.lr, args.batchSz)
    if args.save: save = '{}-{}'.format(args.save, save)
    save = os.path.join('logs', save)

    os.makedirs(save, exist_ok=True)

    print_header('Loading data')

    with open('data/sudoku4data/all_poss.txt', 'r') as f:
        all_sol_string = f.read()
    with open('data/sudoku4data/unique_pos.txt', 'r') as f2:
        all_masks_string = f2.read()

    all_sols = ast.literal_eval(all_sol_string)
    all_sols = np.array(all_sols, dtype='int32')
    all_masks = ast.literal_eval(all_masks_string)

    quiz_data = []
    sol_data = []
    for pos, mask in all_masks:
        # put in solution
        sol_data.append(all_sols[pos].reshape(4, 4))
        # multiply the pos-th sol with the mask to get the puzzle
        s = str(bin(mask))[2:].zfill(16)
        l = [eval(e) for e in s]
        l.reverse()
        m = np.array(l, dtype='int32')
        puzzle = np.multiply(all_sols[pos], m)
        quiz_data.append(puzzle.reshape(4, 4))

    quiz_data = np.array(quiz_data)
    sol_data = np.array(sol_data)

    torch_quiz = torch.from_numpy(quiz_data)
    torch_quiz = F.one_hot(torch_quiz.flatten().long())
    torch_sol = torch.from_numpy(sol_data)
    torch_sol = F.one_hot(torch_sol.flatten().long())
    quizzes = torch_quiz.view(85632, 4, 4, -1).float()
    X_in = quizzes[:, :, :, 1:]

    sols = torch_sol.view(85632, 4, 4, -1).float()
    Y_in = sols[:, :, :, 1:]

    N = X_in.size(0)
    print("number of all data:", N)
    nTrain = int(N * (1. - args.testPct))
    # assert(nTrain % args.batchSz == 0)
    # assert(nTest % args.testBatchSz == 0)

    print_header('Forming inputs')

    X, Y, is_input = process_inputs(X_in, Y_in, 2)

    if args.cuda: X, is_input, Y = X.cuda(), is_input.cuda(), Y.cuda()
    unperm = None

    train_set = TensorDataset(X[:nTrain], is_input[:nTrain], Y[:nTrain])
    test_set = TensorDataset(X[nTrain:], is_input[nTrain:], Y[nTrain:])

    print_header('Building model')
    if args.mnist:
        model = MNISTSudokuSolver(args.boardSz, args.model, args.aux, args.m)
    else:
        model = SudokuSolver(args.boardSz, args.model, args.aux, args.m)

    if args.cuda: model = model.cuda()

    if args.pretrained is not None:
        print(model.load_state_dict(torch.load(args.pretrained)))

    if args.mnist:
        optimizer = optim.Adam([
            {'params': model.sudoku_solver.parameters(), 'lr': args.lr},
            {'params': model.digit_convnet.parameters(), 'lr': 1e-5},
        ])
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_logger = CSVLogger(os.path.join(save, 'train.csv'))
    test_logger = CSVLogger(os.path.join(save, 'test.csv'))
    fields = ['epoch', 'loss', 'err']
    train_logger.log(fields)
    test_logger.log(fields)
    cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    log_file = os.path.join(save, f'{cur_time}.txt')

    old_C = model.C.data.clone()
    for epoch in range(1, args.nEpoch + 1):
        train(args.boardSz, epoch, model, optimizer, train_logger, train_set, args.batchSz, unperm, log_file)
        if args.sparse and torch.norm(model.C.data - old_C) < 0.01 and torch.count_nonzero(model.C.data) > model.C.data.numel() / 2:
            nonzero_idx = torch.nonzero(model.C.data, as_tuple=True)
            min_v = torch.min(torch.abs(model.C.data[nonzero_idx]))
            mask = (torch.abs(model.C.data) != min_v)
            model.C.data = model.C.data * mask

        test(args.boardSz, epoch, model, optimizer, test_logger, test_set, args.testBatchSz, unperm, log_file)
        torch.save(model.state_dict(), os.path.join(save, 'it' + str(epoch) + '.pth'))
        old_C = model.C.data.clone()

        torch.save(model.state_dict(), os.path.join(save, 'model.pth'))


def process_inputs(X, Y, boardSz):
    is_input = X.sum(dim=3, keepdim=True).expand_as(X).int().sign()

    X = X.reshape(X.size(0), -1)
    Y = Y.reshape(Y.size(0), -1)
    is_input = is_input.view(is_input.size(0), -1)

    return X, Y, is_input


def run(boardSz, epoch, model, optimizer, logger, dataset, batchSz, to_train=False, unperm=None, log_file=None):
    loss_final, err_final = 0, 0
    loader = DataLoader(dataset, batch_size=batchSz, shuffle=to_train)
    tloader = tqdm(enumerate(loader), total=len(loader))
    abs_err = 0

    for i, (data, is_input, label) in tloader:
        preds = model(data.contiguous(), is_input.contiguous())

        loss = nn.functional.binary_cross_entropy(preds, label)
        abs_err += torch.abs((preds.data.round() - label)).sum()

        if to_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        err = computeErr(preds.data, boardSz, unperm) / batchSz

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

    # print('memory: {:.2f} MB, cached: {:.2f} MB'.format(torch.cuda.memory_allocated()/2.**20, torch.cuda.memory_cached()/2.**20))
    torch.cuda.empty_cache()


def train(args, epoch, model, optimizer, logger, dataset, batchSz, unperm=None, log_file=None):
    run(args, epoch, model, optimizer, logger, dataset, batchSz, True, unperm, log_file=log_file)


@torch.no_grad()
def test(args, epoch, model, optimizer, logger, dataset, batchSz, unperm=None, log_file=None):
    run(args, epoch, model, optimizer, logger, dataset, batchSz, False, unperm, log_file=log_file)


@torch.no_grad()
def computeErr(pred_flat, n, unperm):
    if unperm is not None: pred_flat[:, :] = pred_flat[:, unperm]

    nsq = n ** 2
    pred = pred_flat.view(-1, nsq, nsq, nsq)

    batchSz = pred.size(0)
    s = (nsq - 1) * nsq // 2  # 0 + 1 + ... + n^2-1

    # slightly cheaping happens here
    # instead of letting each bit prediction round to 0 or 1 independently
    # it leverages domain knowledge to pick only one of [1, 2, .., n]
    I = torch.max(pred, 3)[1].squeeze().view(batchSz, nsq, nsq)

    def invalidGroups(x):
        valid = (x.min(1)[0] == 0)
        valid *= (x.max(1)[0] == nsq - 1)
        valid *= (x.sum(1) == s)
        return valid.bitwise_not()

    boardCorrect = torch.ones(batchSz).type_as(pred)

    # non-cheating
    pred_clip = torch.where(pred > 0.4, 1., 0.)
    for j in range(batchSz):
        if not torch.all(pred_clip[j].sum(-1) == 1.):
            boardCorrect[j] = 0

    for j in range(nsq):
        # Check the jth row and column.
        boardCorrect[invalidGroups(I[:, j, :])] = 0
        boardCorrect[invalidGroups(I[:, :, j])] = 0

        # Check the jth block.
        row, col = n * (j // n), n * (j % n)
        M = invalidGroups(I[:, row:row + n, col:col + n].contiguous().view(batchSz, -1))
        boardCorrect[M] = 0

        if boardCorrect.sum() == 0:
            return batchSz

    return float(batchSz - boardCorrect.sum())


if __name__ == '__main__':
    main()