appliances = ["hvac","fridge","wm","dw","ec","mw","oven","wh"]
features = ["Static", "Monthly+Static", "Monthly"]
appliances=['fridge,"hvac']

import sys
CLUSTER=True
if CLUSTER:
    sys.path.insert(0, '/if6/nb2cz/anaconda/lib/python2.7/site-packages')

sys.path.insert(0, '../code')


import numpy as np
import pandas as pd


from create_df_larger import read_df_larger
df, dfc, all_homes, appliance_min, national_average = read_df_larger()

df = df.rename(columns={'house_num_rooms':'num_rooms',
                        'num_occupants':'total_occupants',
                        'difference_ratio_min_max':'ratio_difference_min_max'})
K_min, K_max = 1,3
F_min, F_max=1,8

from all_functions import *
from features_larger import *





SLURM_OUT = "../slurm_out"
from subprocess import Popen


for feature in features:
    for appliance in appliances:
        for home in all_homes[appliance]:

            OFILE = "%s/%d_%s_%s.out" % (SLURM_OUT, home, appliance, feature)
            EFILE = "%s/%d_%s_%s.err" % (SLURM_OUT, home, appliance,  feature)
            SLURM_SCRIPT = "%d_%s_%s.pbs" % (home, appliance,feature)
            CMD = 'python ../code/main_result_parallel_new_larger.py %s %s %d' % (appliance,feature, home)
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
            Popen(command)
            print appliance, home, feature
