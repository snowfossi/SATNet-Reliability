import argparse
import torch
from os import system
from uniqs_soduku4 import *
import datetime

def construct_maxsat_assignment(index):
    h_assign = all_sols[index]
    assert len(h_assign) == 16
    res = [False] * 64
    for (i, v) in enumerate(h_assign):
        res[4 * i + (v-1)] = True
    
    assign = []
    for i in range(64):
        v = i + 2
        assign.append( v if res[i] else -v)

    return assign

def construct_lowlevel_mask (h_mask):
    i = h_mask
    res = []
    while i > 0:
        res.extend( [i%2==1] * 4)
        i = i // 2

    n = len(res)
    if n < 64:
        res.extend([False] * (64 - n))

    return res

def round(St, thresh):
    cls = []
    for line in St:
        cl = []
        for j,e in enumerate(line):
            if abs(e) >= thresh:
                if e > 0:
                    cl.append(str(j+1))
                else:
                    cl.append(str(-1*(j+1)))
        cls.append(cl)
                
    return cls     

def w_common(cls, f):
    with open(f, 'w') as fout:
        # truth direction must be positive
        fout.write('h 1 0\n')
        for line in cls:
            l = '1 '+ ' '.join(line) + ' 0\n'
            fout.write(l)


def solve(fcom, fpuzzle, fsol, pos, mask):
    assign = construct_maxsat_assignment(pos)
    m = construct_lowlevel_mask (mask)

    with open(fpuzzle, 'a') as fpuz:
        for x in assign:
            v = abs(x)
            if m[v-2]:
                if x > 0:
                    fpuz.write( f'h {v} 0\n' )
                else:
                    fpuz.write( f'h -{v} 0\n' )

    status = system(f'cat {fcom} >> {fpuzzle}')
    if status != 0:
        print("fail to append fcom to fpuzzle")
        return ""
    
    status = system(f'./cashwmaxsatcoreplus -m {fpuzzle} > {fsol}')
    if status != 0:
        print("failed to run maxsat solver on fpuzzle")
        return ""
    
def getsolved(f):
    solved = [0 for i in range(64)]
    with open(f, 'r') as fin:
        for l in fin.read().splitlines():
            if l.startswith('x'):
                result = l.split()
                x_v = result[0].split('x')[1]
                if 1 < int(x_v) < 66:
                    solved[int(x_v)-2] = int(result[1])
    return solved

def compare(solved, gt):
    for i in range(64):
        if solved[i] != gt[i]:
            return 0
    return 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smatrix', type=str)
    parser.add_argument('--aux', type=int, default=0)
    parser.add_argument('--fcom', type=str, default='common.txt')
    parser.add_argument('--fpuzzle', type=str, default='puzzle.txt')
    parser.add_argument('--fsol', type=str, default='maxsat_sol.txt')

    args = parser.parse_args()

    smatrix = torch.load(args.smatrix, map_location=torch.device('cpu'))['S']

    itr = torch.abs(torch.flatten(smatrix))
    
    thresh_sam = itr.tolist()
    
    l = len(uniq_ps)
    start = int(0.9 * l)
    test_ind = [i for i in range(start,l)]

    max_acc = 0
    for thresh in thresh_sam:
        # iterate through threshold value
        correct = 0
        cls = round(smatrix.transpose(0,1), thresh)
        # generate common constraints
        w_common(cls, args.fcom)
        for j in test_ind:
            # iterate through each puzzle in the test set
            pos, mask = uniq_ps[j]
            solve(args.fcom, args.fpuzzle, args.fsol, pos, mask)
            solved = getsolved(args.fsol)
            assign = construct_maxsat_assignment(pos)
            gt = [0 if assign[k] < 0 else 1 for k in range(64)]       
            correct += compare(solved, gt)
        acc = correct / len(test_ind)
        if acc > max_acc:
            max_acc = acc

    print(max_acc)


if __name__ == '__main__':
    print("start time:",datetime.datetime.now())
    main()
    print("end time:", datetime.datetime.now())