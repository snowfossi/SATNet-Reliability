from os import system

for i in range(288):
    system(f'python3 solve_gt.py --pos {i}')

# distribution of weights in the space
'''48 560
  96 562
  96 563
  48 565'''
