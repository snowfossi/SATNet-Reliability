import argparse

def R(fpath):
    with open(fpath, "r") as fin:
        return fin.read().splitlines()

def print_pbo(pbo_cons,g,filename):
    goal = "min: "

    for ele in g:
        (w, var) = ele
        goal = goal + ('+' if w > 0 else '') +str(w) + ' ' + var + ' '
    
    goal = goal + ';'

    with open(filename, 'w') as f: 
        f.write(goal+'\n')

        for p in pbo_cons:
            ls, w = p
            str_reps = [('+' if w > 0 else '') + str(w) + ' ' + name for (w,name) in ls]
            f.write(f"{' '.join(str_reps) } >= {w};"+'\n')


def max_to_pbo(fin, fout):
    # turn maxsat into pbo

    hard_cons = []
    soft_cons = []

    for l in R(fin):
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

    print_pbo(pbo_cons,g,fout)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fin", type=str)
    parser.add_argument("--fout", type=str)

    args=parser.parse_args()
    max_to_pbo(args.fin, args.fout)

if __name__ == "__main__":
    main()
