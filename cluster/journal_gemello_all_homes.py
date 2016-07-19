import sys

CLUSTER = True
if CLUSTER:
    sys.path.insert(0, '/if6/nb2cz/anaconda/lib/python2.7/site-packages')

import numpy as np
import pandas as pd
import pickle
from collections import OrderedDict

out_overall = pickle.load(open('../data/input/all_regions.pkl', 'r'))

region = "Austin"

df = out_overall[region]
df = df.rename(columns={'house_num_rooms': 'num_rooms',
                                  'num_occupants': 'total_occupants',
                                  'difference_ratio_min_max': 'ratio_difference_min_max'})

df = df[(df.full_agg_available == 1) & (df.md_available == 1)]

appliances = ["hvac","fridge","wm","dw","mw","oven"]
features = ["Monthly+Static"]
SLURM_OUT = "../slurm_out"
from subprocess import Popen

import time

for feature in features:
    for appliance in appliances:


        if appliance=="hvac":
            start, stop=5, 11
        else:
            start, stop=1, 13

        appliance_df= df.ix[df[['%s_%d' %(appliance,month) for month in range(start,stop)]].dropna().index]
        for home in appliance_df.index:
            print home, appliance, feature
            OFILE = "%s/%s_%s_%d.out" % (SLURM_OUT, appliance, feature, home)
            EFILE = "%s/%s_%s_%d.err" % (SLURM_OUT, appliance,  feature, home)
            SLURM_SCRIPT = "%s_%s_%d.pbs" % (appliance,feature, home)
            CMD = 'python ../code/journal_gemello_all_homes_leave_one_out_without_feature_optimisation.py %s %s %d' % (appliance,feature, home)
            print CMD
            lines = []
            lines.append("#!/bin/sh\n")
            lines.append('#SBATCH --time=0-06:0:00\n')
            lines.append('#SBATCH --mem=16\n')
            lines.append('#SBATCH -o '+'"' +OFILE+'"\n')
            lines.append('#SBATCH -e '+'"' +EFILE+'"\n')
            lines.append(CMD+'\n')

            with open(SLURM_SCRIPT, 'w') as f:
               f.writelines(lines)
            command = ['sbatch', SLURM_SCRIPT]
            print Popen(command)
            time.sleep(10)
        time.sleep(10)
