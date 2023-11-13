import numpy as np
from os import system
import pickle
import itertools
import argparse

def info(f):
    with open(f, 'rb') as fin:
        d = pickle.load(fin)
    # print(d['description'])
    return d['label']

def convert(configs):
    converted = []
    truth = []
    for ele in configs:
        con = []
        ele_list = ele[0].tolist()
        truth.append(ele_list)
        for i in range(len(ele_list)):
            c = 1*(i+2) if ele_list[i] > 0 else -1 * (i+2)
            con.append(c)
        converted.append(con)
    return converted, truth

def all_configs(converted):
    all = []
    for item in converted:
        for l in range(1,len(item)):
            for comb in itertools.combinations(item, l):
                all.append(list(comb))
    return all

def test(cmatrix, f_poss, aux, fname, n):
    status = system(f'python3 mix2sat.py --cmatrix {cmatrix} --aux {aux} --fout {fname}_com.txt')
    if status != 0: 
        print("fail to generate common")
        return ""

    configs = info(f_poss)
    if fname == "add":
        for i in range(len(configs)):
            configs[i] = [configs[i]]
    converted, truth = convert(configs)

    valid_configs = all_configs(converted)
    for fixed in valid_configs:
        with open(f'fix_{fname}.txt', 'w') as fout:
            for e in fixed:
                fout.write(f'h {e} 0\n')
        status = system(f'cat {fname}_com.txt >> fix_{fname}.txt')
        if status != 0:
            print("fail to concat")
            return ""

        status = system(f'python3 max2pbo.py --fin fix_{fname}.txt --fout fix_{fname}.opb')
        if status != 0:
            print("fail to turn maxsat to pbo")
            return ""
        status = system(f'gurobi_cl Resultfile=fix_{fname}.sol fix_{fname}.opb >fix_{fname}.log')
        if status != 0:
            print("fail to solve using gurobi")
            return ""

        with open(f'fix_{fname}.sol', 'r') as fsol:
            solved = [0 for i in range(n)]
            for l in fsol.read().splitlines():
                for i in range(n):
                    if l.startswith(f'x_{i+2}'):
                        solved[i] = float(l.split(' ')[1])
            if solved not in truth:
                print("wrong solution solved!")
                return ""
    
    print("Great! All solved solutions are valid ground truth!")
     


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmatrix", type=str)
    parser.add_argument("--data", type=str)
    parser.add_argument("--aux", type=int)
    parser.add_argument("--name", type=str)
    parser.add_argument("--n", type=int)
    args = parser.parse_args()

    test(args.cmatrix, args.data, args.aux, args.name, args.n)


if __name__ == "__main__":
    main()