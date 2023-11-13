import numpy as np


def sudoku(boardSize):
    # Define clauses
    clauses = []
    num_choices = boardSize ** 2 # 4
    num_literals = num_choices ** 3 # 64

    # Each cell contains at least one number
    for i in range(num_choices):
        for j in range(num_choices):
            clause = np.zeros(num_literals)
            for k in range(num_choices):
                clause[k + j * num_choices + i * num_choices * num_choices] = 1
            clauses.append(clause)

    # Each cell contains at most one number
    for i in range(num_choices):
        for j in range(num_choices):
            for k1 in range(num_choices):
                for k2 in range(k1 + 1, num_choices):
                    clause = np.zeros(num_literals)
                    clause[k1 + j * num_choices + i * num_choices * num_choices] = -1
                    clause[k2 + j * num_choices + i * num_choices * num_choices] = -1
                    clauses.append(clause)

    # Each row contains each number at most once
    for i in range(num_choices):
        for k in range(num_choices):
            for j1 in range(num_choices):
                for j2 in range(j1 + 1, num_choices):
                    clause = np.zeros(num_literals)
                    clause[k + j1 * num_choices + i * num_choices * num_choices] = -1
                    clause[k + j2 * num_choices + i * num_choices * num_choices] = -1
                    clauses.append(clause)

    # Each column contains each number at most once
    for j in range(num_choices):
        for k in range(num_choices):
            for i1 in range(num_choices):
                for i2 in range(i1 + 1, num_choices):
                    clause = np.zeros(num_literals)
                    clause[k + j * num_choices + i1 * num_choices * num_choices] = -1
                    clause[k + j * num_choices + i2 * num_choices * num_choices] = -1
                    clauses.append(clause)

    # Each subgrid contains each number at most once
    for k in range(num_choices):
        for i1 in range(num_choices):
            for j1 in range(num_choices):
                box_row = (i1 // boardSize) * boardSize
                box_col = (j1 // boardSize) * boardSize
                for i2 in range(i1 + 1, box_row + boardSize):
                    for j2 in range(box_col, box_col + boardSize):
                        if j2 == j1:
                            continue
                        clause = np.zeros(num_literals)
                        clause[k + j1 * num_choices + i1 * num_choices * num_choices] = -1
                        clause[k + j2 * num_choices + i2 * num_choices * num_choices] = -1
                        clauses.append(clause)
    cnf = []
    for row in clauses:
        nonzero_indices = [int((i + 2) * x) for i, x in enumerate(row) if x != 0]
        cnf.append(nonzero_indices)
    with open(f'sudoku{num_choices}_cnf.txt', 'w') as f:
        print(cnf, file=f)

    clauses = np.stack(clauses)
    clauses = np.concatenate((-1 * np.ones((clauses.shape[0], 1)), clauses), axis=1)
    norm = np.sqrt(4 * np.sum(np.abs(clauses), axis=1, keepdims=True))
    clauses /= norm
    return clauses

sudoku4 = sudoku(2)
sudoku9 = sudoku(3)
np.save('sudoku4_gt_rules.npy', sudoku4.T)
np.save('sudoku9_gt_rules.npy', sudoku9.T)