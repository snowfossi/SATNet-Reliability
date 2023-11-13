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


def R(fpath):
    with open(fpath, "r") as fin:
        return fin.read().splitlines()

def print_pbo(pbo_cons,g,filename):
    goal = "min: "

    for ele in g:
        (w, var) = ele
        goal = goal + ('+' if w > 0 else '') +str(w) + ' ' + var + ' '
    
    goal = goal + ';'

    with open(filename, 'w') as fout: 
        fout.write(goal+'\n')


        for p in pbo_cons:
            ls, w = p
            str_reps = [('+' if w > 0 else '') + str(w) + ' ' + name for (w,name) in ls]
            fout.write(f"{' '.join(str_reps) } >= {w};"+'\n')

    

def max_to_pbo(r, filename, pos):
    # turn maxsat into pbo, but with partial assignment

    hard_cons = []
    soft_cons = []

    for l in R(f"cplusdelta_com.txt"):
        if l.startswith('h'):
            vs = list(map(int, l[1:-1].split()))
            hard_cons.append(vs)
        else:
            vs = list(map(int, l[:-1].split()))
            soft_cons.append( (vs[0], vs[1:]) )


    pbo_cons = []
    g =[]
    
    ''' hard constraint example:
    h -1 -2 66 0 
    => 
    -1 x1 -1 x2 +1 66 >= -1'''

    for h in hard_cons:
        
        ct = len(list(filter(lambda x : x < 0, h)))
        xs = [f"x{abs(x)}" for x in h]
        ks = [abs(x)//x for x in h]
        res = list(zip(ks, xs))

        pbo_cons.append((res, 1-ct))


    ''' soft constraint example:
    12622 -66 0
    =>
    let z0 == -66 
    then in the objective, we put + 12622 z0
    instead of 12622*(-x66)'''

    for i, v in enumerate(soft_cons):
        (w, s) = v

        ct = len(list(filter(lambda x : x < 0, s)))
        xs = [f"x{abs(x)}" for x in s]
        ks = [abs(x)//x * w * -1 for x in s]
        res = list(zip(ks, xs))
        # we have +12622 x66 here (to be minimized in the objective)
        g.append(res[0])



    # fix the partial assignment bits
    p_sol = construct_maxsat_assignment(pos)
    # p_mask = construct_lowlevel_mask(mask)

    for x in p_sol:
        v = abs(x)
        if x > 0:
            pbo_cons.append( ([(1, f'x{v}')], 1) )
        else:
            pbo_cons.append( ([(-1, f'x{v}')], 0) )


    # has to be one-hot (i.e. exactly 1 non-zero in four consecutive bis)
    # constraints = [[2+j+i for i in range(4)] for j in range(0,64,4)]
    # for ele in constraints:
        
    #     pbo_cons.append( ([(-1, f'x{ele[k]}') for k in range(4)], -1))
    #     pbo_cons.append( ([(1, f'x{ele[k]}') for k in range(4)], 1))


    print_pbo(pbo_cons,g,filename)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pos', type=int)
    args = parser.parse_args()

    # obtain ground truth solution
    max_to_pbo(0, f'./pbo/gt_pbo_{args.pos}.opb', args.pos)

    status = system(f'gurobi_cl ResultFile=./sol/gt_{args.pos}.sol ./pbo/gt_pbo_{args.pos}.opb > ./glog/g_{args.pos}.log')
    if status != 0:
        print("gurobi: failed to run gurobi solver")
        return ""
                



if __name__=="__main__":
    main()
