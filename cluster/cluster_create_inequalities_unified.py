import time



import pandas as pd
import pickle
import os
import numpy as np
SLURM_OUT = "../slurm_out"
from subprocess import Popen
import time

print "a"

out_overall = pickle.load(open('../data/input/all_regions.pkl','r'))
num_trials=5

print "b"
K = 3
min_num_features=0
train_region="SanDiego"
test_region="Austin"
best_transformation = {appliance:["None","None-percentage"] for appliance in ["dw","mw","wm","oven"]}
best_transformation = {'fridge':'DD',
                           'hvac':'DD-percentage',
                           'wm':'regional',
                           'mw':'DD-percentage',
                           'dw':'None',
                           'oven':'DD'}
for appliance, transformation_list in best_transformation.iteritems():
    for transform in transformation_list:
        for num_homes in range(4, 40, 2):
            OFILE = "%s/%d_%s_%s_%s_%s.out" % (SLURM_OUT, num_homes, train_region[0], test_region[0],appliance[0], transform[0] )
            EFILE = "%s/%d_%s_%s_%s_%s.err" % (SLURM_OUT, num_homes, train_region[0], test_region[0],  appliance,  transform )

            SLURM_SCRIPT = "%d_%s_%s_%s_%s.pbs" % (num_homes, train_region[0], test_region[0],  appliance[:2], transform)
            CMD = 'python ../code/unified_all_homes.py %s %s %s %s %d %d %d %d' % (train_region, test_region,
                                                                                             appliance,
                                                                                             transform, K, num_homes, num_trials, min_num_features)
            lines = []
            lines.append("#!/bin/sh\n")
            lines.append('#SBATCH --time=2-05:0:00\n')
            lines.append('#SBATCH --mem=16\n')
            lines.append('#SBATCH -o '+'"' +OFILE+'"\n')
            lines.append('#SBATCH -e '+'"' +EFILE+'"\n')
            lines.append(CMD+'\n')

            with open(SLURM_SCRIPT, 'w') as f:
               f.writelines(lines)
            command = ['sbatch', SLURM_SCRIPT]
            Popen(command)
            print appliance, transform, num_homes
            import time
            time.sleep(20)
        time.sleep(200)
    time.sleep(1200)


