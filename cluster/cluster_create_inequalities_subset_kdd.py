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
num_trials=25

print "b"
K = 3
for train_region in ["SanDiego"]:
    if train_region=="Austin":
        NUM_HOMES_MAX = 45
    elif train_region=="SanDiego":
        NUM_HOMES_MAX = len(out_overall['SanDiego'])
    else:
        NUM_HOMES_MAX = len(out_overall['Boulder'])

    NUM_HOMES_MAX=20
    for test_region in ["Austin"]:
        if train_region!=test_region:
            TRANSFORMATIONS = ["None","DD","DD-percentage","median-aggregate-percentage",
                              "median-aggregate",'regional','regional-percentage']
        else:
            TRANSFORMATIONS = ["None"]


        train_df = out_overall[train_region]
        test_df = out_overall[test_region]
        test_df=test_df[(test_df.full_agg_available==1)&(test_df.md_available==1)]


        NUM_HOMES_MIN=4

        for num_homes in range(NUM_HOMES_MIN, NUM_HOMES_MAX, 2):


            for transform in TRANSFORMATIONS:
            #for transform in ["None","DD","DD-percentage"]:
            #for transform in ["median-aggregate-percentage"]:
                print transform
                print "*"*40
                count = 0
                #for appliance in ["dw",'hvac','fridge','wm','mw','ec','wh','oven']:
                for appliance in ["hvac"]:
                    if appliance=="hvac":
                        month_min, month_max = 5, 11
                    else:
                        month_min, month_max = 1, 13
                    count+= 1

                    #for appliance in ["hvac","fridge","dr","wm"]:

                    test_df = test_df.ix[test_df[['%s_%d' %(appliance,month) for month in range(month_min, month_max)]].dropna().index]
                    for test_home in test_df.index:
                    #for appliance in ["mw"]:


                        if len(test_df.ix[test_home][['%s_%d' %(appliance, m) for m in range(month_min, month_max)]].dropna())==0:
                            # Appliance data not present for this homes..let's save some time
                            continue

                        print appliance, test_home, count, len(test_df.index), K, transform, train_region, test_region

                        OFILE = "%s/%d_%s_%s_%d_%s_%s.out" % (SLURM_OUT, num_homes, train_region[0], test_region[0], test_home, appliance[0], transform[0] )
                        EFILE = "%s/%d_%s_%s_%d_%s_%s.err" % (SLURM_OUT, num_homes, train_region[0], test_region[0], test_home, appliance,  transform )

                        SLURM_SCRIPT = "%d_%s_%s_%d_%s_%s.pbs" % (num_homes, train_region[0], test_region[0], test_home, appliance[:2], transform)
                        CMD = 'python ../new_experiments/create_inequalities_subset_kdd.py %s %s %d %s %s %d %d %d' % (train_region, test_region,
                                                                                                         test_home, appliance,
                                                                                                         transform, K, num_homes, num_trials)
                        lines = []
                        lines.append("#!/bin/sh\n")
                        lines.append('#SBATCH --time=0-05:0:00\n')
                        lines.append('#SBATCH --mem=16\n')
                        lines.append('#SBATCH -o '+'"' +OFILE+'"\n')
                        lines.append('#SBATCH -e '+'"' +EFILE+'"\n')
                        lines.append(CMD+'\n')

                        with open(SLURM_SCRIPT, 'w') as f:
                           f.writelines(lines)
                        command = ['sbatch', SLURM_SCRIPT]
                        Popen(command)
                        #os.remove(SLURM_SCRIPT)
                print "Now sleeping.."
                import time
                time.sleep(40)
            time.sleep(400)
        time.sleep(1200)


