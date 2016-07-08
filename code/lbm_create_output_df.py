import os, pickle
import pandas as pd
out_path = os.path.expanduser("~/wiki_15min_output_lbm/")

out_overall = pickle.load(open('../data/input/all_regions.pkl', 'r'))
region = "San Diego"
region = region.replace(" ","")
whole=False
df_region = out_overall[region]
APPLIANCES= ["hvac","fridge","wm","mw","dw","oven"]

if region=="Austin" and whole==True:
    df_region = df_region[(df_region.full_agg_available == 1) & (df_region.md_available == 1)]

out = {}
for home in df_region.index:

    try:
        df = pd.read_csv("/if6/nb2cz/wiki_15min_output_lbm/%d.csv" %home,index_col=0)
        df.index = pd.to_datetime(df.index)
        df_res = df.resample("1M",how="sum")
        df_res_kwh = df_res.mul(0.000017).mul(15)
        months = df_res_kwh.index.month
        df_res_kwh.index = months
        out[home]={}
        for month in months:
            for appliance in APPLIANCES:
                out[home]['%s_%d' %(appliance, month)]=df_res_kwh.loc[month, appliance]



    except Exception, e:
        print e
out_df = pd.DataFrame(out).T
pickle.dump(out_df,open(os.path.expanduser('~/lbm_15_%s.pkl' %region),'w'))

