SLURM_OUT = "../slurm_out"


import pickle

out_overall = pickle.load(open('../data/input/all_regions.pkl', 'r'))
region = "Austin"
df = out_overall[region]

from subprocess import Popen

homes =df.index

for home in homes:
    OFILE = "%s/%d.out" % (SLURM_OUT,home )
    EFILE = "%s/%d.err" % (SLURM_OUT, home)

    SLURM_SCRIPT = "%d.pbs" % (home)
    CMD = 'python ../code/run_lbm.py %d' % (home)
    print CMD
    lines = []
    lines.append("#!/bin/sh\n")
    lines.append('#SBATCH --time=1-05:0:00\n')
    lines.append('#SBATCH --mem=16\n')
    lines.append('#SBATCH -o '+'"' +OFILE+'"\n')
    lines.append('#SBATCH -e '+'"' +EFILE+'"\n')
    lines.append(CMD+'\n')

    with open(SLURM_SCRIPT, 'w') as f:
       f.writelines(lines)
    command = ['sbatch', SLURM_SCRIPT]
    Popen(command)
