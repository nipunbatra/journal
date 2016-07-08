from latent_Bayesian_melding import LatentBayesianMelding
lbm = LatentBayesianMelding()
meterlist = ['hvac','fridge','dw','wm','mw','oven']

import pickle
import os



lbm_file = '../data/input/austin_15min.json'
data_path = os.path.expanduser("~/wiki_15min_mains/")
out_path = os.path.expanduser("~/wiki_15min_output_lbm/")

individual_model = lbm.import_model(meterlist,lbm_file)

import pandas as pd
import warnings
import sys
warnings.filterwarnings("ignore")

home, = sys.argv[1:]
home = int(home)



df = pd.read_csv("%s/%d.csv" %(data_path, home), index_col=0, names=['localminute', 'use'])
df.index = pd.to_datetime(df.index)

df['day'] = df.index.dayofyear
g = df.groupby("day")
out = []
for day in range(1, 365):
    try:
        progress = day*100.0/365
        print("%0.2f done" %progress)
        sys.stdout.flush()
        mains = g.get_group(day)['use']
        result = lbm.disaggregate_chunk(mains)
        out.append(result['inferred appliance energy'])
    except Exception, e:
        print e
res_home = pd.concat(out)
#return res_home
res_home.to_csv("%s%d.csv" %(out_path,home))


#disaggregate_home(home)
