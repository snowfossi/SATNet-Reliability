import os
import argparse
import subprocess
from uniqs_soduku4 import uniq_ps
from concurrent.futures.process import ProcessPoolExecutor



class Generator:
    def __init__(self):
        self.exec_dir = os.getcwd()
    
    def run(self, process):
        position = uniq_ps[process][0]
        mask = uniq_ps[process][1]

        cmd_line = ["python3", "g_solve_puzzle.py", "--pos", str(position), "--mask", str(mask)]

        try:
            process = subprocess.Popen(cmd_line, cwd=self.exec_dir, start_new_session=True)
            out, err = process.communicate()
    

        except Exception as e:
            print("Error: ", e)
          

        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_process', type=int, default=8, help='Number of processes to run in parallel')
    parser.add_argument('--start', type=int, default=0, help='which position to start')
    parser.add_argument('--length', type=int, default=1)
    

    opts = parser.parse_args()

    generator = Generator()
    
    all_runs = [i+opts.start for i in range(opts.length)]
    with ProcessPoolExecutor(max_workers=opts.n_process) as pool:
        pool.map(generator.run, all_runs)
    

if __name__ == '__main__':
    main()
