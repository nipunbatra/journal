import os
import pandas as pd

test_region = "Austin"
START_SEED =0
STOP_SEED = 25

base_path = os.path.expanduser("~/output/unified/kdd_all_features/")
out_mean = {}
out_sem ={}
for appliance in ["fridge","hvac","wm"]:
    out_mean[appliance]={}
    out_sem[appliance]={}
    for feature in ["Monthly","Monthly+Static","Static"]:
        out_mean[appliance][feature]={}
        out_sem[appliance][feature]={}
        for NUM_HOMES in range(4, 60, 2):
            try:
                file_path = os.path.join(base_path, "%s_%d_%d_%d_%s.csv" %(appliance, NUM_HOMES, START_SEED, STOP_SEED, feature))
                df = pd.read_csv(file_path, header=None,index_col=0)
                out_mean[appliance][feature][NUM_HOMES] = df.mean()
                out_sem[appliance][feature][NUM_HOMES] = df.sem()
            except:
                pass




