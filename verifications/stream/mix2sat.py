import argparse
import torch
import numpy as np

INF_W = 2 ** 30

name_to_ids = {}

class Lit:
    def __init__(self, pos, name):
        self.pos = pos
        self.name = name
    
    def __repr__(self) -> str:
        return self.name if self.pos else f'(not {self.name})'

class Clause:
    def __init__(self, lits=[]):
        self.lits = lits

    def add_lit(self, pos, name):
        self.lits.append( (pos, name))
    
    def __repr__(self) -> str:
        return f'{self.lits}'

class SATInstance:
    def __init__(self, cls=[]):
        self.cls = cls
    def add_clause(self, w, c):
        self.cls.append( (w, c) )
    def __repr__(self) -> str:
        return "SatInstance with {0} clauses:\n{1}".format(len(self.cls),'\n'.join([f'{w} {c}' for w,c in self.cls]) )

    def save_as_maxsat(self, fpath):
        global name_to_ids
        v = len(name_to_ids)
        for _, c in self.cls:
            for lit in c.lits:
                if lit.name not in name_to_ids:
                    v += 1
                    name_to_ids[lit.name] = v

        with open(fpath, 'w') as fout:
            # output the name to id mapping (optional)
            # for x in name_to_ids:
            #     fout.write(f'c {x} ==> {name_to_ids[x]}\n')

            # output weighted clauses

            # truth vector must be positive
            fout.write("h 1 0\n")
            for w, c in self.cls:
                fout.write( 'h' if w == INF_W else f'{w}')
                for lit in c.lits:
                    fout.write( f' {"" if lit.pos else "-"}{name_to_ids[lit.name]}')
                fout.write(' 0\n')

def encode_eq(s, vi, vj, w):
    eq_name = f'Eq_{vi}_{vj}'

    if w == 0:
        pass
    elif w > 0:
        s.add_clause(INF_W, Clause([Lit(False, vi), Lit(True, vj), Lit(False, eq_name)]))
        s.add_clause(INF_W, Clause([Lit(True, vi), Lit(False, vj), Lit(False, eq_name)]))
        s.add_clause(w, Clause([Lit(True, eq_name)]))
        pass
    elif w < 0:
        s.add_clause(INF_W, Clause([Lit(False, vi), Lit(False, vj), Lit(True, eq_name)]))
        s.add_clause(INF_W, Clause([Lit(True, vi), Lit(True, vj), Lit(True, eq_name)]))
        s.add_clause(-w, Clause([Lit(False, eq_name)]))


def interpret_C_matrix(cmatrix, aux):    
    s = (cmatrix.transpose(0, 1) - cmatrix).sum()
    if s > 1e-2:
        print('cmatrix should be symmetric')
        exit()

    n = cmatrix.size(0)
    names = [ f'v{i}' for i in range(n - aux) ]
    names.extend([f'a{i}' for i in range(aux)])

    v = len(name_to_ids)
    for x in names:
        v += 1
        name_to_ids[x] = v

    sat = SATInstance()

    for i in range(n):
        for j in range(i):
            w = -1 * cmatrix[i][j].item()
            # for streams 
            w = int(30*w)
            encode_eq(sat, names[i], names[j], w)

    return sat
 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cmatrix', type=str)
    parser.add_argument('--aux', type=int, default=0)
    parser.add_argument('--fout', type=str, default='w.txt')

    args = parser.parse_args()

    cmatrix = torch.from_numpy(np.load(args.cmatrix))
    sat = interpret_C_matrix(cmatrix, args.aux)
    sat.save_as_maxsat(args.fout)


if __name__ == '__main__':
    main()