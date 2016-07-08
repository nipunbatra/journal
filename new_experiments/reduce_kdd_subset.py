appliance="hvac"
feature="Monthly+Static"
start_seed, end_seed = 0,25
out_mean = {}
out_sem = {}
out_median ={}
import pandas as pd
for homes in range(4, 60, 2):
    print homes
    path = "/if6/nb2cz/output/unified/kdd_all_features/%s_%d_%d_%d_%s.csv" %(appliance,
                                                                            homes,
                                                                             start_seed,
                                                                             end_seed,
                                                                             feature)
    df = pd.read_csv(path, index_col=0, header=None).squeeze()
    out_mean[homes] = df.mean()
    out_median[homes]=df.median()
    out_sem[homes] = df.sem()



