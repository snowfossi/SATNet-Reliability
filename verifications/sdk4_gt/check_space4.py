from os import system
import argparse
from uniqs_soduku4 import all_sols


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



def main():
    for i in range(288):
        assign = construct_maxsat_assignment(i)
        with open('full.txt', 'w') as fout:
            for e in assign:
                fout.write(f'h {e} 0\n')
        status = system('cat common4.txt >> full.txt')
        system('./cashwmaxsatcoreplus -m full.txt >> sdk4_sol.txt')
        with open('sdk4_sol.txt','r') as fin:
            for l in fin.read().splitlines():
                if l.startswith('o'):
                    w = l
            print(w)

if __name__ == "__main__":
    main()