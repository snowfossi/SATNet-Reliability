import pickle
import numpy as np
from os import system
import time


with open('sudoku9_err_instances.pickle', 'rb') as f:
    d = pickle.load(f)
dt_pred = d['output']
dt_truth = d['label']

for i in range(len(dt_pred)):
    predicted = np.zeros(1029)
    gt = np.zeros(1029)

    for j in range(1,1030):
        predicted[j-1] = j+1 if dt_pred[i][0][j] > 0.5 else -1*(j+1)
        # aux vars
        if j > 729:
            gt[j-1] = predicted[j-1]
    
    
    for k in range(729):
        gt[k] = k+2 if dt_truth[i][0][k] > 0 else -1*(k+2)
    
    
    with open('full9_pred.txt', 'w') as fout:
        for e in predicted:
            fout.write(f'h {int(e)} 0\n')
    system('cat mix_common9.txt >> full9_pred.txt')
    
    with open('full9_gt_a.txt', 'w') as f2:
        for e in gt:
            f2.write(f'h {int(e)} 0\n')
    system('cat mix_common9.txt >> full9_gt_a.txt')

    system('python3 max2pbo.py --fin full9_pred.txt --fout pbo9pred.opb')
    system('python3 max2pbo.py --fin full9_gt_a.txt --fout pbo9gt_a.opb')

    system('gurobi_cl Resultfile=pbo9pred.sol pbo9pred.opb')
    system('gurobi_cl Resultfile=pbo9gt_a.sol pbo9gt_a.opb')
    with open('pbo9pred.sol', 'r') as f1:
        pred_obj = int(f1.readline().split(' ')[4].split('\n')[0])
        result = f"testset {i} pred obj: {pred_obj}, "

    with open('pbo9gt_a.sol', 'r') as f2:
        gt_obj = int(f2.readline().split(' ')[4].split('\n')[0])
        result += f"testset {i} gt obj: {gt_obj}, "
    
    opt = gt_obj < pred_obj
    print(result + f"gt more optimal than pred: {opt}\n", file=open('out_a.txt', 'a'))


    # time.sleep(30)

    # print(dt_truth[i][0])




    
