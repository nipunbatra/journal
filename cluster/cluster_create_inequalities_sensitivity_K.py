import time



import pandas as pd
import pickle

SLURM_OUT = "../slurm_out"
from subprocess import Popen
import time

out_overall = pickle.load(open('../data/input/all_regions.pkl','r'))

train_region = "SanDiego"
test_region = "Austin"
best_transform = {}

best_transform['SanDiego']={'fridge': 'DD', 'hvac': 'DD-percentage', 'wm': 'median-aggregate-percentage'}
#for test region
best_transform['Austin']= {'fridge':'DD',
                           'hvac':'DD-percentage',
                           'wm':'regional',
                           'mw':'DD-percentage',
                           'dw':'None',
                           'oven':'DD'}

train_df = out_overall[train_region]
test_df = out_overall[test_region]
K=3
#for K in [1,2,4,5,6,7]:
for K in [6,7]:
    #for appliance in ["hvac","fridge","wm","dw","mw","oven"]:
    for appliance in ["hvac","wm"]:
        transform = best_transform[test_region][appliance]

        print transform
        print "*"*40
        count = 0
        for test_home in test_df.index:
            count+= 1


            print appliance, test_home, count, len(test_df.index), K, transform

            OFILE = "%s/%s_%s_%d_%s_%s.out" % (SLURM_OUT, train_region, test_region, test_home, appliance,transform )
            EFILE = "%s/%s_%s_%d_%s_%s.err" % (SLURM_OUT, train_region, test_region, test_home, appliance, transform )

            SLURM_SCRIPT = "%s_%s_%d_%s_%s.pbs" % (train_region, test_region, test_home, appliance,  transform)
            CMD = 'python ../new_experiments/create_inequalities.py %s %s %d %s %s %d' % (train_region, test_region,
                                                                                             test_home, appliance,
                                                                                             transform, K)
            lines = []
            lines.append("#!/bin/sh\n")
            lines.append('#SBATCH --time=0-01:0:00\n')
            lines.append('#SBATCH --mem=16\n')
            lines.append('#SBATCH -o '+'"' +OFILE+'"\n')
            lines.append('#SBATCH -e '+'"' +EFILE+'"\n')
            lines.append(CMD+'\n')

            with open(SLURM_SCRIPT, 'w') as f:
               f.writelines(lines)
            command = ['sbatch', SLURM_SCRIPT]
            Popen(command)
            print "Now sleeping.."
            #time.sleep(1)
        #time.sleep(3*60)


