import pandas as pd
import glob
import pickle

out = {}
for appliance in ["hvac","fridge","wm","mw","dw","oven"]:
    print appliance
    if appliance not in out:
        out[appliance]={}
    for feature in ["Monthly+Static"]:
    #for feature in ["Monthly","Static","Monthly+Static"]:
        print appliance, feature
        #out[appliance][feature]={}


        f = glob.glob("%s_%s*.pkl" %(appliance, feature))

        for g in f:
            out[appliance][int(g.split("_")[2][:-4])] = pickle.load(open(g,'r'))['error'].squeeze()

res = {}
for appliance in ["hvac","fridge","wm","mw","dw","oven"]:
    res[appliance]=pd.DataFrame(out[appliance]).T.median().mean()

import pandas as pd
import glob
import pickle

out = {}
for appliance in ["hvac","fridge","wm","mw","dw","oven"]:
    out[appliance]={}



    f = glob.glob("Austin_%s*.pkl" %(appliance))

    for g in f:
        out[appliance][int(g.split("_")[2])] = pickle.load(open(g,'r'))['error'].squeeze()

res_vis = {}
for appliance in ["hvac","fridge","wm","mw","dw","oven"]:
    res_vis[appliance]=pd.DataFrame(out[appliance]).T.median().mean()