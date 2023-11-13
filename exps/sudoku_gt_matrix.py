import torch

gid = 1


def get_id():
    global gid
    gid += 1
    assert gid > 1  # 1 is reserved for truth
    return gid


# when sat,  r = w
# when unsat, r = 0
def encode_a_b_c(a, b, c, d):
    # assume w = 2
    w = 2
    r = 0

    if a == b: r += -w / 2
    if a == c: r += -w / 2
    if b == c: r += -w / 2

    if a == True: r += w / 2
    if b == True: r += w / 2
    if c == True: r += w / 2
    if d == True: r += -w / 2

    if a == d: r += w / 2
    if b == d: r += w / 2
    if c == d: r += w / 2

    return r


# when sat,  r = w
# when unsat, r = 0
def encode_nota_b_c(a, b, c, d):
    # assume w = 2
    w = 20
    r = 0

    if a == b: r += w / 2
    if a == c: r += w / 2
    if b == c: r += -w / 2

    if a == True: r += -w / 2
    if b == True: r += w / 2
    if c == True: r += w / 2
    if d == True: r += -w / 2

    if a == d: r += -w / 2
    if b == d: r += w / 2
    if c == d: r += w / 2

    return r


# when sat,  r = 0
# when unsat, r = -w
def encode_nota_notb_c(a, b, c, d):
    # assume w = 2
    w = 2
    r = 0

    if a == b: r += -w / 2
    if a == c: r += w / 2
    if b == c: r += w / 2

    if a == True: r += -w / 2
    if b == True: r += -w / 2
    if c == True: r += w / 2
    if d == True: r += -w / 2

    if a == d: r += -w / 2
    if b == d: r += -w / 2
    if c == d: r += w / 2

    return r


# when sat,  r = -2w
# when unsat, r = -3w
def encode_nota_notb_notc(a, b, c, d):
    # assume w = 2
    w = 2
    r = 0

    if a == b: r += -w / 2
    if a == c: r += -w / 2
    if b == c: r += -w / 2

    if a == True: r += -w / 2
    if b == True: r += -w / 2
    if c == True: r += -w / 2
    if d == True: r += -w / 2

    if a == d: r += -w / 2
    if b == d: r += -w / 2
    if c == d: r += -w / 2

    return r


def test(encoding_f):
    vs = [True, False]
    for a in vs:
        for b in vs:
            for c in vs:
                r1 = encoding_f(a, b, c, True)
                r2 = encoding_f(a, b, c, False)
                print(f'a={a}, b={b}, c={c}, max weight: {max(r1, r2)}')


def encode_two_lit(w, cl):
    assert w % 2 == 0
    assert len(cl) == 2

    a, b = cl
    if a > 0 and b > 0:
        return (-w, w, w)

    if a > 0 and b < 0:
        return (w / 2, w / 2, -w / 2)

    if a < 0 and b > 0:
        return (w / 2, -w / 2, w / 2)

    if a < 0 and b < 0:
        return (-w / 2, -w / 2, -w / 2)


def my_update(res, d):
    for k in d:
        if k in res:
            res[k] += d[k]
        else:
            res[k] = d[k]


def encode_two_lit(w, cl, res):
    assert w % 2 == 0
    assert len(cl) == 2
    for x in cl:
        assert abs(x) > 1

    a, b = cl
    if a > 0 and b > 0:
        x = min(a, b)
        y = max(a, b)
        my_update(res, {(x, y): -w / 2, (1, x): w / 2, (1, y): w / 2})

    if a > 0 and b < 0:
        x = min(a, -b)
        y = max(a, -b)

        my_update(res, {(x, y): w / 2, (1, a): w / 2, (1, -b): -w / 2})

    if a < 0 and b > 0:
        x = min(-a, b)
        y = max(-a, b)

        my_update(res, {(x, y): w / 2, (1, -a): -w / 2, (1, b): w / 2})

    if a < 0 and b < 0:
        x = min(-a, -b)
        y = max(-a, -b)
        my_update(res, {(x, y): -w / 2, (1, -a): -w / 2, (1, -b): -w / 2})

    return res


def encode_three_lit_0(w, cl):
    assert len(cl) == 3
    assert w % 2 == 0
    for x in cl:
        assert abs(x) > 1

    a, b, c = cl

    d = get_id()
    res = {}
    if a > 0 and b > 0 and c > 0:
        res.update({
            (min(a, b), max(a, b)): -w / 2,
            (min(a, c), max(a, c)): -w / 2,
            (min(b, c), max(b, c)): -w / 2,

            (1, a): w / 2,
            (1, b): w / 2,
            (1, c): w / 2,
            (1, -d): -w / 2,

            (min(a, d), max(a, d)): w / 2,
            (min(b, d), max(b, d)): w / 2,
            (min(c, d), max(c, d)): w / 2,
        })

    if a > 0 and b > 0 and c < 0:
        pass


def encode_three_lit(w, cl, res):
    # w = 2
    assert len(cl) == 3
    assert w % 2 == 0
    for x in cl:
        assert abs(x) > 1

    a, b, c = cl

    d = get_id()

    for x in cl:
        if x > 0:
            my_update(res, {(1, x): w})
        else:
            my_update(res, {(1, -x): -w})

    my_update(res, {(1, d): w})

    encode_two_lit(w, [-a, -b], res)
    encode_two_lit(w, [-b, -c], res)
    encode_two_lit(w, [-a, -c], res)

    encode_two_lit(w, [a, -d], res)
    encode_two_lit(w, [b, -d], res)
    encode_two_lit(w, [c, -d], res)


def sudoku23sat(formula):
    res = []
    var_num = 65
    for clause in formula:
        if len(clause) <= 3:
            res.append(clause)
        else:
            assert len(clause) == 4
            var_num += 1
            c = var_num
            res.append([clause[0], clause[1], c])
            res.append([clause[2], clause[3], -c])

    print('var_num', var_num)
    return res


def encode_S_matrix(n, r):
    s = []
    for cl in r:
        t = [0 for _ in range(n)]
        t[0] = -1.0
        factor = (4 * len(cl)) ** 0.5
        for x in cl:
            v = abs(x) - 1
            t[v] = (1.0 if x > 0 else -1.0) / factor

        s.append(t)

    S_matrix = torch.tensor(s) * -1
    torch.save(S_matrix, "gt4_S_matrix.pt")
    print(f'S_matrix size: {S_matrix.size()}')


def encode_C_matrix(n, r):
    pass

    lit4 = 0
    for cl in r:
        if len(cl) == 4:
            lit4 += 1

    rules = sudoku23sat(r)
    for clause in rules:
        if len(clause) == 3:
            encode_three_lit(20, (clause[0], clause[1], clause[2]), res)
        else:
            assert len(clause) == 2
            encode_two_lit(20, [clause[0], clause[1]], res)

    c = []
    n = n + lit4 * 3  # each lit-4 clause needs 3 auxilliary variables (one for breaking into two 3-lit clauses, and each 3-lit clause requires an aux variable)
    for i in range(1, n + 1):
        row = []
        for j in range(1, n + 1):
            if i == j:
                row.append(0)
            else:
                x = min(i, j)
                y = max(i, j)
                if (x, y) in res:
                    row.append(res[(x, y)])
                else:
                    print(f'cannot find a weight for {(x, y)}')
                    row.append(0)
        c.append(row)

    C_matrix = torch.tensor(c) * -1
    torch.save(C_matrix, "../verifications/w_extra_info/delta_c_matrix_one_hot_rules.pt")
    print(f"n = {n}, gid = {gid}")
    print(c)


if __name__ == '__main__':
    res = {}

    r = [[2, 3, 4, 5], [6, 7, 8, 9], [10, 11, 12, 13], [14, 15, 16, 17], [18, 19, 20, 21], [22, 23, 24, 25], [26, 27, 28, 29], [30, 31, 32, 33], [34, 35, 36, 37], [38, 39, 40, 41], [42, 43, 44, 45], [46, 47, 48, 49], [50, 51, 52, 53], [54, 55, 56, 57], [58, 59, 60, 61], [62, 63, 64, 65], [-2, -3], [-2, -4], [-2, -5], [-3, -4], [-3, -5], [-4, -5], [-6, -7], [-6, -8], [-6, -9], [-7, -8], [-7, -9], [-8, -9], [-10, -11], [-10, -12], [-10, -13], [-11, -12], [-11, -13], [-12, -13], [-14, -15], [-14, -16], [-14, -17], [-15, -16], [-15, -17], [-16, -17], [-18, -19], [-18, -20], [-18, -21], [-19, -20], [-19, -21], [-20, -21], [-22, -23], [-22, -24], [-22, -25], [-23, -24], [-23, -25], [-24, -25], [-26, -27], [-26, -28], [-26, -29], [-27, -28], [-27, -29], [-28, -29], [-30, -31], [-30, -32], [-30, -33], [-31, -32], [-31, -33], [-32, -33], [-34, -35], [-34, -36], [-34, -37], [-35, -36], [-35, -37], [-36, -37], [-38, -39], [-38, -40], [-38, -41], [-39, -40], [-39, -41], [-40, -41], [-42, -43], [-42, -44], [-42, -45], [-43, -44], [-43, -45], [-44, -45], [-46, -47], [-46, -48], [-46, -49], [-47, -48], [-47, -49], [-48, -49], [-50, -51], [-50, -52], [-50, -53], [-51, -52], [-51, -53], [-52, -53], [-54, -55], [-54, -56], [-54, -57], [-55, -56], [-55, -57], [-56, -57], [-58, -59], [-58, -60], [-58, -61], [-59, -60], [-59, -61], [-60, -61], [-62, -63], [-62, -64], [-62, -65], [-63, -64], [-63, -65], [-64, -65], [-2, -6], [-2, -10], [-2, -14], [-6, -10], [-6, -14], [-10, -14], [-3, -7], [-3, -11], [-3, -15], [-7, -11], [-7, -15], [-11, -15], [-4, -8], [-4, -12], [-4, -16], [-8, -12], [-8, -16], [-12, -16], [-5, -9], [-5, -13], [-5, -17], [-9, -13], [-9, -17], [-13, -17], [-18, -22], [-18, -26], [-18, -30], [-22, -26], [-22, -30], [-26, -30], [-19, -23], [-19, -27], [-19, -31], [-23, -27], [-23, -31], [-27, -31], [-20, -24], [-20, -28], [-20, -32], [-24, -28], [-24, -32], [-28, -32], [-21, -25], [-21, -29], [-21, -33], [-25, -29], [-25, -33], [-29, -33], [-34, -38], [-34, -42], [-34, -46], [-38, -42], [-38, -46], [-42, -46], [-35, -39], [-35, -43], [-35, -47], [-39, -43], [-39, -47], [-43, -47], [-36, -40], [-36, -44], [-36, -48], [-40, -44], [-40, -48], [-44, -48], [-37, -41], [-37, -45], [-37, -49], [-41, -45], [-41, -49], [-45, -49], [-50, -54], [-50, -58], [-50, -62], [-54, -58], [-54, -62], [-58, -62], [-51, -55], [-51, -59], [-51, -63], [-55, -59], [-55, -63], [-59, -63], [-52, -56], [-52, -60], [-52, -64], [-56, -60], [-56, -64], [-60, -64], [-53, -57], [-53, -61], [-53, -65], [-57, -61], [-57, -65], [-61, -65], [-2, -18], [-2, -34], [-2, -50], [-18, -34], [-18, -50], [-34, -50], [-3, -19], [-3, -35], [-3, -51], [-19, -35], [-19, -51], [-35, -51], [-4, -20], [-4, -36], [-4, -52], [-20, -36], [-20, -52], [-36, -52], [-5, -21], [-5, -37], [-5, -53], [-21, -37], [-21, -53], [-37, -53], [-6, -22], [-6, -38], [-6, -54], [-22, -38], [-22, -54], [-38, -54], [-7, -23], [-7, -39], [-7, -55], [-23, -39], [-23, -55], [-39, -55], [-8, -24], [-8, -40], [-8, -56], [-24, -40], [-24, -56], [-40, -56], [-9, -25], [-9, -41], [-9, -57], [-25, -41], [-25, -57], [-41, -57], [-10, -26], [-10, -42], [-10, -58], [-26, -42], [-26, -58], [-42, -58], [-11, -27], [-11, -43], [-11, -59], [-27, -43], [-27, -59], [-43, -59], [-12, -28], [-12, -44], [-12, -60], [-28, -44], [-28, -60], [-44, -60], [-13, -29], [-13, -45], [-13, -61], [-29, -45], [-29, -61], [-45, -61], [-14, -30], [-14, -46], [-14, -62], [-30, -46], [-30, -62], [-46, -62], [-15, -31], [-15, -47], [-15, -63], [-31, -47], [-31, -63], [-47, -63], [-16, -32], [-16, -48], [-16, -64], [-32, -48], [-32, -64], [-48, -64], [-17, -33], [-17, -49], [-17, -65], [-33, -49], [-33, -65], [-49, -65], [-2, -22], [-6, -18], [-10, -30], [-14, -26], [-34, -54], [-38, -50], [-42, -62], [-46, -58], [-3, -23], [-7, -19], [-11, -31], [-15, -27], [-35, -55], [-39, -51], [-43, -63], [-47, -59], [-4, -24], [-8, -20], [-12, -32], [-16, -28], [-36, -56], [-40, -52], [-44, -64], [-48, -60], [-5, -25], [-9, -21], [-13, -33], [-17, -29], [-37, -57], [-41, -53], [-45, -65], [-49, -61]]

    lit = []
    for cl in r:
        if len(cl) == 2:
            lit.append(cl)
    print(lit)

    encode_C_matrix(65, lit)
