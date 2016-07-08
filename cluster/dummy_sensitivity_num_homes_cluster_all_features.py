import os

appliances = ["hvac","fridge","wm","mw","dw","oven"]
appliances = ['hvac','fridge']
features = ["Static","Monthly","Monthly+Static"]
features=["Monthly+Static","Static","Monthly"]

SLURM_OUT = "../slurm_out"
if not os.path.exists(SLURM_OUT):
    os.makedirs(SLURM_OUT)
from subprocess import Popen
appliance="hvac"
START_SEED =0
STOP_SEED = 25
NUM_HOMES=4
feature = "Static"
OFILE = "%s/A_%sN_%dS_%d_F%s.out" % (SLURM_OUT, appliance, NUM_HOMES, START_SEED, feature)
EFILE = "%s/A_%sN_%dS_%d_F%s.err" % (SLURM_OUT, appliance, NUM_HOMES, START_SEED, feature)
SLURM_SCRIPT = "A_%sN_%dS_%d_F%s.pbs" % (appliance, NUM_HOMES, START_SEED, feature)
CMD = 'python ../code/kdd_final.py %s %d %d %d %s' % (appliance,NUM_HOMES,START_SEED,STOP_SEED,feature)
lines = []
lines.append("#!/bin/sh\n")
lines.append('#SBATCH --time=1-02:0:00\n')
lines.append('#SBATCH --mem=16\n')
lines.append('#SBATCH -o '+'"' +OFILE+'"\n')
lines.append('#SBATCH -e '+'"' +EFILE+'"\n')
lines.append(CMD+'\n')

with open(SLURM_SCRIPT, 'w') as f:
   f.writelines(lines)
command = ['sbatch', SLURM_SCRIPT]
print Popen(command)
