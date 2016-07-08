appliances = ["hvac","fridge","wm","mw","dw","oven"]
#appliances = ['hvac','fridge']
features = ["Static","Monthly","Monthly+Static"]
features=["Monthly+Static","Static","Monthly"]
features=["Monthly+Static"]
appliances= ["hvac"]
SLURM_OUT = "../slurm_out"
from subprocess import Popen
import time
for appliance in appliances:
    START_SEED =0
    STOP_SEED = 25

    for NUM_HOMES in range(8, 40, 2):
        #time.sleep(60)
        for feature in features:
            OFILE = "%s/A_%sN_%dS_%d_F%s.out" % (SLURM_OUT, appliance, NUM_HOMES, START_SEED, feature)
            EFILE = "%s/A_%sN_%dS_%d_F%s.err" % (SLURM_OUT, appliance, NUM_HOMES, START_SEED, feature)
            SLURM_SCRIPT = "A_%sN_%dS_%d_F%s.pbs" % (appliance, NUM_HOMES, START_SEED, feature)
            CMD = 'python ../code/kdd_final_percentage.py %s %d %d %d %s' % (appliance,NUM_HOMES,START_SEED,STOP_SEED,feature)
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
