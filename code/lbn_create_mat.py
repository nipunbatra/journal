import warnings
import os
warnings.filterwarnings('ignore')
import pandas as pd
metadata_df = pd.read_csv("../data/input/dataport-metadata.csv",index_col=0)

city = "Austin"
time_period = "15T"
APPLIANCE_MAP = {"hvac":'air1',
                 'fridge':'refrigerator1',
                 'wm':'clotheswasher1',
                 'dw':'dishwasher1',
                 'mw':'microwave1',
                 'oven':'oven1'
                 }
data_path = os.path.expanduser("~/wiki_15min/")

sd_data = metadata_df[metadata_df['city'] == city]
sd_homes = sd_data.index.values.astype('int')

out_df = {}
for appliance_name_code in APPLIANCE_MAP.keys():
    out_df[appliance_name_code] = []


mat_appliance={}
for home in sd_homes[:]:
    try:
        home_df = pd.read_csv(os.path.join(data_path,"%d.csv" %home), index_col=['local_15min'])
        home_df.index = pd.to_datetime(home_df.index)
        home_df = home_df['2014']
        home_df = home_df*1000
        for appliance_name_code, appliance_name in APPLIANCE_MAP.iteritems():
            print home, appliance_name_code
            if appliance_name in home_df.columns:
                if appliance_name_code=="hvac":
                    df = home_df[appliance_name]['2014-5-1':'2014-10-31']
                else:
                    df = home_df[appliance_name]
                try:
                    df = df.resample(time_period).astype("float64").head(96*30)
                    df = df.groupby(level=0).last()
                    df = pd.DataFrame(df)
                    df.columns=['power']
                    df["day"] = df.index.dayofyear
                    df["minute"] = df.index.hour*60 + df.index.minute
                    df = df.pivot(index="day", values="power",columns="minute")
                    out_df[appliance_name_code].append(df)
                except:
                    pass
    except Exception, e:
        print e


for appliance_name_code in APPLIANCE_MAP.iterkeys():
    mat_appliance[appliance_name_code] = pd.concat(out_df[appliance_name_code]).values

